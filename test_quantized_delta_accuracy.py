"""
Test Model Accuracy with Quantized Delta Compression

This script evaluates whether quantized delta compression affects actual model performance
on downstream tasks. It compresses checkpoints, reconstructs them, and measures task accuracy.

Usage:
    # Test all quantization methods on a specific task
    python test_quantized_delta_accuracy.py --base-checkpoint run_final_inc_rank/continual/task_0_C-STANCE --target-checkpoint run_final_inc_rank/continual/task_1_FOMC --dataset FOMC

    # Test specific quantization type only
    python test_quantized_delta_accuracy.py --base-checkpoint <path> --target-checkpoint <path> --dataset FOMC --quantization-type int8

    # Test with larger eval set
    python test_quantized_delta_accuracy.py --base-checkpoint <path> --target-checkpoint <path> --dataset FOMC --num-eval-samples 500
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Add src to path for evaluation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from evaluations import (
    eval_ScienceQA,
    eval_MeetingBank,
    eval_CStance,
    eval_Py150,
    eval_FOMC,
    eval_NumGLUE_cm,
    eval_NumGLUE_ds,
    eval_20Minuten
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class QuantizedDeltaTester:
    """Tests model accuracy with quantized delta compression."""

    def __init__(
        self,
        base_checkpoint: str,
        target_checkpoint: str,
        base_model_name: str = "openlm-research/open_llama_3b_v2",
        device: str = "cuda",
        use_4bit_base: bool = True,
    ):
        """
        Initialize the tester.

        Args:
            base_checkpoint: Path to base checkpoint (e.g., task_0)
            target_checkpoint: Path to target checkpoint (e.g., task_1)
            base_model_name: HuggingFace model identifier
            device: Device to run inference on
            use_4bit_base: Whether to load base model with 4-bit quantization
        """
        self.base_checkpoint_path = Path(base_checkpoint)
        self.target_checkpoint_path = Path(target_checkpoint)
        self.base_model_name = base_model_name
        self.device = device
        self.use_4bit_base = use_4bit_base

        # Load checkpoint weights
        logger.info(f"Loading base checkpoint: {self.base_checkpoint_path}")
        self.base_weights = load_file(
            self.base_checkpoint_path / "adapter_model.safetensors"
        )

        logger.info(f"Loading target checkpoint: {self.target_checkpoint_path}")
        self.target_weights = load_file(
            self.target_checkpoint_path / "adapter_model.safetensors"
        )

        self.tokenizer = None
        self.base_model = None

    def setup_tokenizer(self):
        """Load tokenizer."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer from {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, use_fast=False
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def setup_base_model(self):
        """Load base model (cached for reuse)."""
        if self.base_model is None:
            logger.info(f"Loading base model: {self.base_model_name}")

            bnb_config = None
            if self.use_4bit_base:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

    def compute_delta(self) -> Dict[str, torch.Tensor]:
        """Compute weight differences between target and base."""
        delta = {}
        for key in self.target_weights.keys():
            if key in self.base_weights:
                delta[key] = self.target_weights[key] - self.base_weights[key]
            else:
                # New weights in target
                delta[key] = self.target_weights[key]
        return delta

    def quantize_delta(
        self, delta: Dict[str, torch.Tensor], dtype: torch.dtype
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Quantize delta weights.

        Args:
            delta: Dictionary of delta tensors
            dtype: Target dtype (torch.float16, torch.bfloat16, or torch.int8)

        Returns:
            Tuple of (quantized_delta, compression_time)
        """
        logger.info(f"Quantizing delta to {dtype}")
        start_time = time.time()

        quantized_delta = {}
        for key, tensor in delta.items():
            if dtype == torch.int8:
                # INT8 quantization with scale and zero point
                tensor_min = tensor.min()
                tensor_max = tensor.max()

                scale = (tensor_max - tensor_min) / 255.0
                zero_point = -torch.round(tensor_min / scale).to(torch.int8)

                quantized = torch.round(tensor / scale + zero_point.float()).clamp(
                    -128, 127
                ).to(torch.int8)

                quantized_delta[key] = quantized
                quantized_delta[f"{key}_scale"] = scale.reshape(1)
                quantized_delta[f"{key}_zero_point"] = zero_point.reshape(1)
            else:
                # FP16/BF16 quantization
                quantized_delta[key] = tensor.to(dtype)

        compression_time = time.time() - start_time
        return quantized_delta, compression_time

    def dequantize_delta(
        self,
        quantized_delta: Dict[str, torch.Tensor],
        original_delta_keys: List[str],
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Dequantize delta weights.

        Args:
            quantized_delta: Quantized delta dictionary
            original_delta_keys: Original delta keys (without scale/zero_point)
            dtype: Original quantization dtype

        Returns:
            Tuple of (dequantized_delta, decompression_time)
        """
        logger.info(f"Dequantizing delta from {dtype}")
        start_time = time.time()

        dequantized = {}
        for key in original_delta_keys:
            if dtype == torch.int8:
                quantized = quantized_delta[key]
                scale = quantized_delta[f"{key}_scale"]
                zero_point = quantized_delta[f"{key}_zero_point"]

                dequantized[key] = (
                    quantized.float() - zero_point.float()
                ) * scale
            else:
                dequantized[key] = quantized_delta[key].float()

        decompression_time = time.time() - start_time
        return dequantized, decompression_time

    def reconstruct_weights(
        self, dequantized_delta: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct target weights from base + dequantized delta.

        Args:
            dequantized_delta: Dequantized delta dictionary

        Returns:
            Reconstructed weights dictionary
        """
        reconstructed = {}
        for key in self.target_weights.keys():
            if key in dequantized_delta:
                if key in self.base_weights:
                    reconstructed[key] = self.base_weights[key] + dequantized_delta[key]
                else:
                    reconstructed[key] = dequantized_delta[key]
            else:
                # Fallback to original if not in delta
                reconstructed[key] = self.target_weights[key]

        return reconstructed

    def load_model_with_weights(
        self, weights: Dict[str, torch.Tensor]
    ) -> PeftModel:
        """
        Load model with given weights.

        Args:
            weights: LoRA adapter weights to load

        Returns:
            Model with loaded weights
        """
        self.setup_base_model()

        # Save weights to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "adapter_model.safetensors"
            save_file(weights, str(tmp_path))

            # Copy adapter config from target checkpoint
            import shutil
            shutil.copy(
                self.target_checkpoint_path / "adapter_config.json",
                Path(tmp_dir) / "adapter_config.json"
            )

            # Load model with weights
            model = PeftModel.from_pretrained(self.base_model, str(tmp_dir))
            model.eval()

        return model

    def evaluate_model(
        self,
        model: PeftModel,
        dataset_name: str,
        num_samples: Optional[int] = None,
        max_ans_len: int = 256,
        temperature: float = 0.1,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            model: Model to evaluate
            dataset_name: Name of the dataset
            num_samples: Number of samples to evaluate (None = all)
            max_ans_len: Maximum answer length
            temperature: Generation temperature
            batch_size: Batch size for inference

        Returns:
            Dictionary with evaluation metrics
        """
        self.setup_tokenizer()

        logger.info(f"Evaluating model on {dataset_name}")

        # Load dataset
        data_dir = Path("TRACE-Benchmark/LLM-CL-Benchmark_5000")
        dataset_path = data_dir / dataset_name

        logger.info(f"Loading test dataset from {dataset_path}")
        dataset = load_dataset(
            "json",
            data_files={"test": str(dataset_path / "test.json")},
        )

        test_data = dataset["test"]
        if num_samples:
            test_data = test_data.select(range(min(num_samples, len(test_data))))

        logger.info(f"Evaluating on {len(test_data)} samples")

        # Generate predictions
        predictions = []
        ground_truths = []
        sources = []

        model.eval()

        from tqdm import tqdm
        for i in tqdm(range(0, len(test_data), batch_size), desc="Generating"):
            batch = test_data[i : i + batch_size]

            # Prepare inputs
            prompts = batch["prompt"]
            answers = batch["answer"]

            sources.extend(prompts)
            ground_truths.extend(answers)

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            prompt_len = inputs.input_ids.shape[1]

            # Generate
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_ans_len,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.unk_token_id,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    use_cache=True,
                )

            # Decode
            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            predictions.extend(sequences)

        # Evaluate predictions
        logger.info("Computing metrics...")
        if dataset_name == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predictions, ground_truths)
        elif dataset_name == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(predictions, ground_truths)
        elif dataset_name == "C-STANCE":
            evaluation_result = eval_CStance.eval(predictions, ground_truths)
        elif dataset_name == "Py150":
            evaluation_result = eval_Py150.eval(predictions, ground_truths)
        elif dataset_name == "FOMC":
            evaluation_result = eval_FOMC.eval(predictions, ground_truths)
        elif dataset_name == "NumGLUE-cm":
            evaluation_result = eval_NumGLUE_cm.eval(predictions, ground_truths)
        elif dataset_name == "NumGLUE-ds":
            evaluation_result = eval_NumGLUE_ds.eval(predictions, ground_truths)
        elif dataset_name == "20Minuten":
            evaluation_result = eval_20Minuten.eval(sources, predictions, ground_truths)
        else:
            logger.warning(f"No evaluation function for {dataset_name}")
            evaluation_result = {}

        return evaluation_result

    def test_quantization_method(
        self,
        dtype: torch.dtype,
        dataset_name: str,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Test a single quantization method.

        Args:
            dtype: Quantization dtype
            dataset_name: Dataset to evaluate on
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with results
        """
        dtype_name = str(dtype).split(".")[-1].upper()
        logger.info("=" * 70)
        logger.info(f"Testing Quantized Delta ({dtype_name})")
        logger.info("=" * 70)

        results = {"method": f"Quantized Delta ({dtype_name})"}

        # Step 1: Compute delta
        logger.info("Step 1: Computing delta...")
        delta = self.compute_delta()

        # Step 2: Quantize delta
        logger.info("Step 2: Quantizing delta...")
        quantized_delta, compression_time = self.quantize_delta(delta, dtype)
        results["compression_time_s"] = compression_time

        # Step 3: Dequantize delta
        logger.info("Step 3: Dequantizing delta...")
        dequantized_delta, decompression_time = self.dequantize_delta(
            quantized_delta, list(delta.keys()), dtype
        )
        results["decompression_time_s"] = decompression_time

        # Step 4: Reconstruct weights
        logger.info("Step 4: Reconstructing target weights...")
        reconstructed_weights = self.reconstruct_weights(dequantized_delta)

        # Step 5: Compute reconstruction error
        max_error = 0.0
        avg_error = 0.0
        total_elements = 0

        for key in self.target_weights.keys():
            if key in reconstructed_weights:
                diff = torch.abs(self.target_weights[key] - reconstructed_weights[key])
                max_error = max(max_error, diff.max().item())
                avg_error += diff.sum().item()
                total_elements += diff.numel()

        avg_error /= total_elements if total_elements > 0 else 1
        results["max_weight_error"] = max_error
        results["mean_weight_error"] = avg_error

        logger.info(f"Max weight error: {max_error:.2e}")
        logger.info(f"Mean weight error: {avg_error:.2e}")

        # Step 6: Load model and evaluate
        logger.info("Step 5: Loading model with reconstructed weights...")
        model = self.load_model_with_weights(reconstructed_weights)

        logger.info("Step 6: Evaluating model accuracy...")
        eval_metrics = self.evaluate_model(
            model, dataset_name, num_samples=num_samples
        )
        results["evaluation_metrics"] = eval_metrics

        logger.info(f"Evaluation metrics: {eval_metrics}")

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache()

        return results

    def test_original_model(
        self, dataset_name: str, num_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test the original (uncompressed) model as baseline.

        Args:
            dataset_name: Dataset to evaluate on
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with results
        """
        logger.info("=" * 70)
        logger.info("Testing Original Model (No Compression)")
        logger.info("=" * 70)

        results = {"method": "Original (No Compression)"}

        # Load model with original weights
        logger.info("Loading model with original weights...")
        model = self.load_model_with_weights(self.target_weights)

        logger.info("Evaluating model accuracy...")
        eval_metrics = self.evaluate_model(
            model, dataset_name, num_samples=num_samples
        )
        results["evaluation_metrics"] = eval_metrics

        logger.info(f"Evaluation metrics: {eval_metrics}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Test Model Accuracy with Quantized Delta Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to base checkpoint (e.g., task_0)",
    )
    parser.add_argument(
        "--target-checkpoint",
        type=str,
        required=True,
        help="Path to target checkpoint (e.g., task_1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to evaluate on (e.g., FOMC, C-STANCE)",
    )
    parser.add_argument(
        "--quantization-type",
        type=str,
        choices=["fp16", "bf16", "int8", "all"],
        default="all",
        help="Quantization type to test (default: all)",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="quantized_delta_accuracy_results.json",
        help="Output JSON file (default: quantized_delta_accuracy_results.json)",
    )
    parser.add_argument(
        "--no-4bit-base",
        action="store_true",
        help="Disable 4-bit quantization for base model (uses more memory)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Quantized Delta Compression - Model Accuracy Test")
    logger.info("=" * 70)
    logger.info(f"Base checkpoint: {args.base_checkpoint}")
    logger.info(f"Target checkpoint: {args.target_checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Quantization type: {args.quantization_type}")
    logger.info(f"Eval samples: {args.num_eval_samples or 'all'}")
    logger.info("=" * 70)

    # Create tester
    tester = QuantizedDeltaTester(
        base_checkpoint=args.base_checkpoint,
        target_checkpoint=args.target_checkpoint,
        use_4bit_base=not args.no_4bit_base,
    )

    all_results = {}

    # Test original model (baseline)
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE: Original Model (No Compression)")
    logger.info("=" * 70)
    original_results = tester.test_original_model(
        args.dataset, num_samples=args.num_eval_samples
    )
    all_results["original"] = original_results

    # Test quantization methods
    quantization_methods = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }

    if args.quantization_type == "all":
        methods_to_test = quantization_methods
    else:
        methods_to_test = {args.quantization_type: quantization_methods[args.quantization_type]}

    for method_name, dtype in methods_to_test.items():
        logger.info("\n" + "=" * 70)
        logger.info(f"TESTING: Quantized Delta ({method_name.upper()})")
        logger.info("=" * 70)

        results = tester.test_quantization_method(
            dtype, args.dataset, num_samples=args.num_eval_samples
        )
        all_results[f"quantized_{method_name}"] = results

    # Generate comparison report
    logger.info("\n" + "=" * 70)
    logger.info("ACCURACY COMPARISON REPORT")
    logger.info("=" * 70)

    # Print comparison table
    print(f"\n{'Method':<30} {'Weight Error (Max)':<20} {'Metrics':<50}")
    print("-" * 100)

    for method_key, result in all_results.items():
        method_name = result["method"]
        weight_error = result.get("max_weight_error", 0.0)
        metrics = result.get("evaluation_metrics", {})

        # Format metrics as string
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())

        print(f"{method_name:<30} {weight_error:<20.2e} {metrics_str:<50}")

    # Save results to JSON
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")

    # Print recommendations
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    print("""
Based on the accuracy evaluation:

1. **FP16 Quantization**: Minimal accuracy loss, ~50% compression
   - Use if accuracy is critical
   - Good balance of compression and quality

2. **BF16 Quantization**: Similar to FP16, better for some architectures
   - Same compression as FP16
   - Better numerical stability in some cases

3. **INT8 Quantization**: Maximum compression (~75%), check accuracy degradation
   - Best compression ratio
   - May have slight accuracy drop - validate carefully
   - Recommended if accuracy loss is < 1-2%

4. **Next Steps**:
   - If accuracy loss is acceptable: Deploy quantized delta compression
   - If accuracy loss is too high: Try hybrid approaches (see suggestions below)
   - Test on more tasks to ensure consistency
    """)


if __name__ == "__main__":
    main()
