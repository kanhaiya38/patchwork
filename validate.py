"""
Standalone Validation Script for Continual Learning

Run validation on trained model checkpoints independently from training.

Usage:
    # Validate a specific checkpoint
    python validate.py --checkpoint-dir ./lora-continual/task_2_MeetingBank

    # Validate all checkpoints
    python validate.py --validate-all

    # Evaluate all checkpoints + base model on a single dataset (measure forgetting)
    python validate.py --eval-all-on-dataset MeetingBank

    # Evaluate only the base model
    python validate.py --eval-base-model MeetingBank
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from evaluation import run_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class EvalDataCollator:
    """Custom data collator for evaluation that preserves source text and labels."""

    def __init__(self, tokenizer, max_prompt_len: int = 512):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __call__(self, examples):
        # Extract prompts and answers
        prompts = [ex['prompt'] for ex in examples]
        answers = [ex['answer'] for ex in examples]

        # Tokenize prompts only (for generation)
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_prompt_len,
            padding='longest',
            return_tensors='pt'
        )

        # Return batch with source text and ground truths preserved
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'sources': prompts,
            'gts': answers
        }


def load_base_model(
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    use_quantization: bool = True,
    device: str = "cuda"
):
    """
    Load the base model without any LoRA adapters (same config as training.py).

    Args:
        base_model_name: Base model identifier
        use_quantization: Whether to use 4-bit quantization (same as training)
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")

    # Configure quantization (same as training.py)
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model (same as training.py)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Base model loaded successfully")
    return model, tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    use_quantization: bool = True,
    device: str = "cuda"
):
    """
    Load model from a LoRA checkpoint (same config as training.py).

    Args:
        checkpoint_path: Path to the LoRA checkpoint
        base_model_name: Base model identifier
        use_quantization: Whether to use 4-bit quantization (same as training)
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")

    # Configure quantization (same as training.py)
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model (same as training.py)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA checkpoint
    logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


def validate_base_model(
    dataset_name: str,
    data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
    output_dir: str = "./validation-results",
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    max_prompt_len: int = 512,
    max_ans_len: int = 256,
    temperature: float = 0.1,
    batch_size: int = 4,
    use_quantization: bool = True,
    device: str = "cuda"
):
    """
    Run validation on the base model (without LoRA) for a specific dataset.

    Args:
        dataset_name: Name of dataset to validate on
        data_dir: Root directory for datasets
        output_dir: Directory to save validation results
        base_model_name: Base model identifier
        max_prompt_len: Maximum prompt length
        max_ans_len: Maximum answer length for generation
        temperature: Generation temperature
        batch_size: Batch size for inference
        use_quantization: Whether to use 4-bit quantization
        device: Device to run on
    """
    logger.info("=" * 80)
    logger.info(f"VALIDATION: BASE MODEL (no LoRA)")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 80)

    # Load base model
    model, tokenizer = load_base_model(
        base_model_name=base_model_name,
        use_quantization=use_quantization,
        device=device
    )

    # Load dataset
    dataset_path = Path(data_dir) / dataset_name
    logger.info(f"Loading dataset from: {dataset_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "eval": str(dataset_path / "eval.json"),
        },
    )

    # Create dataloader
    eval_collator = EvalDataCollator(tokenizer, max_prompt_len=max_prompt_len)
    eval_sampler = SequentialSampler(dataset["eval"])
    eval_dataloader = DataLoader(
        dataset["eval"],
        collate_fn=eval_collator,
        sampler=eval_sampler,
        batch_size=batch_size
    )

    # Run validation (task_id = -1 for base model)
    evaluation_result = run_validation(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        dataset_name=dataset_name,
        task_id=-1,  # Special ID for base model
        output_path=output_dir,
        max_ans_len=max_ans_len,
        temperature=temperature,
        device=device
    )

    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Results: {evaluation_result}")
    logger.info("=" * 80)

    return evaluation_result


def validate_checkpoint(
    checkpoint_path: str,
    dataset_name: str,
    data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
    output_dir: str = "./validation-results",
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    max_prompt_len: int = 512,
    max_ans_len: int = 256,
    temperature: float = 0.1,
    batch_size: int = 4,
    use_quantization: bool = True,
    device: str = "cuda"
):
    """
    Run validation on a single checkpoint for a specific dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_name: Name of dataset to validate on
        data_dir: Root directory for datasets
        output_dir: Directory to save validation results
        base_model_name: Base model identifier
        max_prompt_len: Maximum prompt length
        max_ans_len: Maximum answer length for generation
        temperature: Generation temperature
        batch_size: Batch size for inference
        use_quantization: Whether to use 4-bit quantization
        device: Device to run on
    """
    logger.info("=" * 80)
    logger.info(f"VALIDATION: {checkpoint_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 80)

    # Load model
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        use_quantization=use_quantization,
        device=device
    )

    # Load dataset
    dataset_path = Path(data_dir) / dataset_name
    logger.info(f"Loading dataset from: {dataset_path}")

    dataset = load_dataset(
        "json",
        data_files={
            "eval": str(dataset_path / "eval.json"),
        },
    )

    # Create dataloader
    eval_collator = EvalDataCollator(tokenizer, max_prompt_len=max_prompt_len)
    eval_sampler = SequentialSampler(dataset["eval"])
    eval_dataloader = DataLoader(
        dataset["eval"],
        collate_fn=eval_collator,
        sampler=eval_sampler,
        batch_size=batch_size
    )

    # Extract task_id from checkpoint path (e.g., task_2_MeetingBank -> 2)
    checkpoint_name = Path(checkpoint_path).name
    try:
        task_id = int(checkpoint_name.split('_')[1])
    except:
        task_id = 0
        logger.warning(f"Could not extract task_id from {checkpoint_name}, using 0")

    # Run validation
    evaluation_result = run_validation(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=eval_dataloader,
        dataset_name=dataset_name,
        task_id=task_id,
        output_path=output_dir,
        max_ans_len=max_ans_len,
        temperature=temperature,
        device=device
    )

    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Results: {evaluation_result}")
    logger.info("=" * 80)

    return evaluation_result


def validate_all_checkpoints(
    checkpoint_base_dir: str = "./lora-continual",
    data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
    output_dir: str = "./validation-results",
    **kwargs
):
    """
    Validate all checkpoints found in the checkpoint directory.

    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        data_dir: Root directory for datasets
        output_dir: Directory to save validation results
        **kwargs: Additional arguments passed to validate_checkpoint
    """
    checkpoint_base = Path(checkpoint_base_dir)

    # Find all checkpoint directories
    checkpoints = sorted([d for d in checkpoint_base.iterdir() if d.is_dir() and d.name.startswith('task_')])

    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_base_dir}")
        return

    logger.info(f"Found {len(checkpoints)} checkpoints")

    # Validate each checkpoint
    for checkpoint_path in checkpoints:
        # Extract dataset name from checkpoint directory name
        # e.g., task_2_MeetingBank -> MeetingBank
        parts = checkpoint_path.name.split('_', 2)
        if len(parts) >= 3:
            dataset_name = parts[2]
        else:
            logger.warning(f"Could not extract dataset name from {checkpoint_path.name}, skipping")
            continue

        try:
            validate_checkpoint(
                checkpoint_path=str(checkpoint_path),
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Validation failed for {checkpoint_path}: {e}")
            continue

    logger.info("=" * 80)
    logger.info("ALL VALIDATIONS COMPLETE")
    logger.info("=" * 80)


def validate_all_on_single_dataset(
    dataset_name: str,
    checkpoint_base_dir: str = "./lora-continual",
    data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
    output_dir: str = "./validation-results",
    include_base_model: bool = True,
    **kwargs
):
    """
    Validate all checkpoints on a single specified dataset.

    This is useful for measuring forgetting - how well each checkpoint
    performs on a specific dataset after training on other tasks.

    Args:
        dataset_name: Name of the dataset to evaluate all checkpoints on
        checkpoint_base_dir: Base directory containing checkpoints
        data_dir: Root directory for datasets
        output_dir: Directory to save validation results
        include_base_model: Whether to also evaluate the base model (default: True)
        **kwargs: Additional arguments passed to validate_checkpoint
    """
    checkpoint_base = Path(checkpoint_base_dir)

    # Find all checkpoint directories
    checkpoints = sorted([d for d in checkpoint_base.iterdir() if d.is_dir() and d.name.startswith('task_')])

    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_base_dir}")
        return

    logger.info(f"Found {len(checkpoints)} checkpoints")
    logger.info(f"Evaluating {'base model + ' if include_base_model else ''}{len(checkpoints)} checkpoints on dataset: {dataset_name}")
    logger.info("=" * 80)

    results_summary = []

    # Evaluate base model first if requested
    if include_base_model:
        logger.info(f"\n{'='*80}")
        logger.info(f"Model: BASE MODEL (no LoRA)")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*80}")

        try:
            evaluation_result = validate_base_model(
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **kwargs
            )

            results_summary.append({
                'checkpoint': 'base_model',
                'dataset': dataset_name,
                'metrics': evaluation_result
            })

        except Exception as e:
            logger.error(f"Validation failed for base model on {dataset_name}: {e}")
            results_summary.append({
                'checkpoint': 'base_model',
                'dataset': dataset_name,
                'metrics': {'error': str(e)}
            })

    # Validate each checkpoint on the same dataset
    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.name
        logger.info(f"\n{'='*80}")
        logger.info(f"Checkpoint: {checkpoint_name}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*80}")

        try:
            evaluation_result = validate_checkpoint(
                checkpoint_path=str(checkpoint_path),
                dataset_name=dataset_name,
                data_dir=data_dir,
                output_dir=output_dir,
                **kwargs
            )

            results_summary.append({
                'checkpoint': checkpoint_name,
                'dataset': dataset_name,
                'metrics': evaluation_result
            })

        except Exception as e:
            logger.error(f"Validation failed for {checkpoint_path} on {dataset_name}: {e}")
            results_summary.append({
                'checkpoint': checkpoint_name,
                'dataset': dataset_name,
                'metrics': {'error': str(e)}
            })
            continue

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 80)

    for result in results_summary:
        logger.info(f"\n{result['checkpoint']}:")
        logger.info(f"  Metrics: {result['metrics']}")

    logger.info("\n" + "=" * 80)
    logger.info("ALL VALIDATIONS COMPLETE")
    logger.info("=" * 80)

    # Save summary to file
    summary_path = Path(output_dir) / f"summary_all_checkpoints_on_{dataset_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific checkpoint on its training dataset
  python validate.py --checkpoint-dir ./lora-continual/task_2_MeetingBank

  # Validate a checkpoint on a different dataset
  python validate.py --checkpoint-dir ./lora-continual/task_2_MeetingBank --dataset ScienceQA

  # Validate all checkpoints on their respective training datasets
  python validate.py --validate-all

  # Evaluate all checkpoints + base model on a single dataset (measure forgetting)
  python validate.py --eval-all-on-dataset MeetingBank

  # Evaluate all checkpoints on a single dataset (exclude base model)
  python validate.py --eval-all-on-dataset MeetingBank --no-base-model

  # Evaluate only the base model on a dataset
  python validate.py --eval-base-model MeetingBank
        """
    )

    # Checkpoint selection
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Path to specific checkpoint directory to validate"
    )
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate all checkpoints on their respective training datasets"
    )
    parser.add_argument(
        "--eval-all-on-dataset",
        type=str,
        metavar="DATASET",
        help="Evaluate all checkpoints on a single specified dataset (useful for measuring forgetting)"
    )
    parser.add_argument(
        "--eval-base-model",
        type=str,
        metavar="DATASET",
        help="Evaluate only the base model (no LoRA) on a specified dataset"
    )
    parser.add_argument(
        "--include-base-model",
        action="store_true",
        default=True,
        help="Include base model evaluation when using --eval-all-on-dataset (default: True)"
    )
    parser.add_argument(
        "--no-base-model",
        action="store_true",
        help="Exclude base model evaluation when using --eval-all-on-dataset"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="./lora-continual",
        help="Base directory containing checkpoints (default: ./lora-continual)"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to validate on (if not specified, uses dataset from checkpoint name)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="TRACE-Benchmark/LLM-CL-Benchmark_500",
        help="Root directory for datasets (default: TRACE-Benchmark/LLM-CL-Benchmark_500)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation-results",
        help="Directory to save validation results (default: ./validation-results)"
    )

    # Model settings
    parser.add_argument(
        "--base-model",
        type=str,
        default="openlm-research/open_llama_3b_v2",
        help="Base model name (default: openlm-research/open_llama_3b_v2)"
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (same as training, default: True)"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    # Generation settings
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=512,
        help="Maximum prompt length (default: 512)"
    )
    parser.add_argument(
        "--max-ans-len",
        type=int,
        default=256,
        help="Maximum answer length (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32, increase to 64+ for H200)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )

    args = parser.parse_args()

    # Validate arguments
    exclusive_options = sum([
        bool(args.validate_all),
        bool(args.checkpoint_dir),
        bool(args.eval_all_on_dataset),
        bool(args.eval_base_model)
    ])

    if exclusive_options > 1:
        parser.error("Can only specify one of: --validate-all, --checkpoint-dir, --eval-all-on-dataset, or --eval-base-model")

    if exclusive_options == 0:
        parser.error("Must specify one of: --validate-all, --checkpoint-dir, --eval-all-on-dataset, or --eval-base-model")

    # Determine quantization setting
    use_quantization = args.use_quantization and not args.no_quantization

    # Common kwargs
    common_kwargs = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "base_model_name": args.base_model,
        "max_prompt_len": args.max_prompt_len,
        "max_ans_len": args.max_ans_len,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "use_quantization": use_quantization,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Run validation
    if args.validate_all:
        validate_all_checkpoints(
            checkpoint_base_dir=args.checkpoint_base_dir,
            **common_kwargs
        )
    elif args.eval_all_on_dataset:
        # Determine if base model should be included
        include_base = args.include_base_model and not args.no_base_model

        validate_all_on_single_dataset(
            dataset_name=args.eval_all_on_dataset,
            checkpoint_base_dir=args.checkpoint_base_dir,
            include_base_model=include_base,
            **common_kwargs
        )
    elif args.eval_base_model:
        validate_base_model(
            dataset_name=args.eval_base_model,
            **common_kwargs
        )
    else:
        # Extract dataset name from checkpoint if not specified
        if args.dataset:
            dataset_name = args.dataset
        else:
            checkpoint_name = Path(args.checkpoint_dir).name
            parts = checkpoint_name.split('_', 2)
            if len(parts) >= 3:
                dataset_name = parts[2]
            else:
                parser.error(f"Could not extract dataset name from {checkpoint_name}, please specify --dataset")

        validate_checkpoint(
            checkpoint_path=args.checkpoint_dir,
            dataset_name=dataset_name,
            **common_kwargs
        )


if __name__ == "__main__":
    main()
