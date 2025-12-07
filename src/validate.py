"""
Standalone Validation Script for Continual Learning

Run validation on trained model checkpoints independently from training.

Results are automatically organized in timestamped folders:
- validate-all: ./validation-results/all_checkpoints_<timestamp>/
- checkpoint-dir: ./validation-results/<checkpoint_name>_<timestamp>/
- eval-all-on-dataset: ./validation-results/<dataset>_<timestamp>/
- eval-base-model: ./validation-results/base_model_<dataset>_<timestamp>/

Usage:
    # Validate a specific checkpoint (validates on current + all previous tasks by default)
    python validate.py --checkpoint-dir ./lora-continual/task_2_MeetingBank

    # Validate all checkpoints (validates each on current + all previous tasks)
    python validate.py --validate-all ./lora-continual

    # Evaluate all checkpoints + base model on a single dataset (measure forgetting)
    python validate.py --eval-all-on-dataset MeetingBank

    # Evaluate only the base model
    python validate.py --eval-base-model MeetingBank
"""

import argparse
import json
import logging
import sys
from datetime import datetime
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

from src.constants import MAX_PROMPT_LEN, MAX_ANS_LEN, DEFAULT_TASK_CONFIGS
from src.data_collator import DataCollator
from src.evaluation import run_validation

# Global configuration: Set to True to use merged models, False for LoRA checkpoints
# - When True: Loads merged models from experiments/merged_models/ directory
# - When False: Loads base model + LoRA adapter from continual/ directory (backward compatible)
USE_MERGED_MODELS = True  # Default: False for backward compatibility

# Task sequence for continual learning (must match training.py)
TASK_SEQUENCE = [
    'C-STANCE',      # task_0
    'FOMC',          # task_1
    'MeetingBank',   # task_2
    'Py150',         # task_3
    'ScienceQA',     # task_4
    'NumGLUE-cm',    # task_5
    'NumGLUE-ds',    # task_6
    '20Minuten',     # task_7
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# NOTE: EvalDataCollator removed - now using DataCollator with inference=True
# This ensures consistent tokenization between training and inference


def load_base_model(
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    use_quantization: bool = False,
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
    use_quantization: bool = False,
    pre_quantized: bool = False,
    device: str = "cuda"
):
    """
    Load model from a checkpoint.

    Behavior depends on USE_MERGED_MODELS global variable:
    - If True: Load merged model directly from checkpoint_path
    - If False: Load base model + LoRA adapter (old behavior)

    Args:
        checkpoint_path: Path to the checkpoint (merged model or LoRA adapter)
        base_model_name: Base model identifier (only used when USE_MERGED_MODELS=False)
        use_quantization: Whether to use 4-bit quantization (same as training)
        pre_quantized: If True, models are already quantized (don't apply quantization_config)
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    if USE_MERGED_MODELS:
        # NEW: Load merged model directly
        logger.info(f"Loading merged model from: {checkpoint_path}")

        # Configure quantization
        bnb_config = None
        if use_quantization and not pre_quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif pre_quantized:
            logger.info("Loading pre-quantized model (skipping quantization config)")

        # Load tokenizer from merged model directory
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Load merged model directly (no LoRA adapter needed)
        if pre_quantized:
            # For pre-quantized models, load without specifying quantization_config
            # The model is already quantized, so we just load it as-is
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        model.eval()

        logger.info("Merged model loaded successfully")
        return model, tokenizer

    else:
        # OLD: Load base model + LoRA adapter (existing behavior)
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


def resolve_checkpoint_path(checkpoint_base_dir: str, checkpoint_name: str) -> str:
    """
    Resolve checkpoint path based on USE_MERGED_MODELS setting.

    Args:
        checkpoint_base_dir: Base directory (e.g., "./lora-continual" or "./experiments")
        checkpoint_name: Checkpoint directory name (e.g., "task_0_MeetingBank")

    Returns:
        Full path to checkpoint
    """
    if USE_MERGED_MODELS:
        # Look in merged_models subdirectory
        # If checkpoint_base_dir is "./lora-continual", convert to "./experiments/merged_models"
        if "lora-continual" in checkpoint_base_dir or "continual" in checkpoint_base_dir:
            # Replace continual with merged_models
            base_dir = Path(checkpoint_base_dir).parent / "merged_models"
        else:
            # Assume checkpoint_base_dir already points to experiments or similar
            base_dir = Path(checkpoint_base_dir) / "merged_models"

        checkpoint_path = base_dir / checkpoint_name
    else:
        # OLD: Use continual directory
        checkpoint_path = Path(checkpoint_base_dir) / checkpoint_name

    logger.info(f"Resolved checkpoint path: {checkpoint_path}")
    return str(checkpoint_path)


def find_checkpoints(checkpoint_base_dir, task_sequence, use_merged_models=True):
    """
    Discover checkpoints for the given task sequence.

    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        task_sequence: List of task names in order
        use_merged_models: If True, look in merged_models subdirectory

    Returns:
        List of tuples: [(task_id, task_name, checkpoint_path), ...]
    """
    # Resolve checkpoint base directory
    if use_merged_models:
        if "lora-continual" in checkpoint_base_dir or "continual" in checkpoint_base_dir:
            checkpoint_base = Path(checkpoint_base_dir).parent / "merged_models"
        else:
            checkpoint_base = Path(checkpoint_base_dir) / "merged_models"
    else:
        checkpoint_base = Path(checkpoint_base_dir)

    if not checkpoint_base.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_base}")

    checkpoints = []

    for task_id, task_name in enumerate(task_sequence):
        checkpoint_name = f"task_{task_id}_{task_name}"
        checkpoint_path = checkpoint_base / checkpoint_name

        if checkpoint_path.exists() and checkpoint_path.is_dir():
            checkpoints.append((task_id, task_name, str(checkpoint_path)))
            logger.info(f"Found checkpoint: {checkpoint_name}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_name}, skipping...")

    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_base} for the given task sequence")

    return checkpoints


def validate_base_model(
    dataset_name: str,
    data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
    output_dir: str = "./validation-results",
    base_model_name: str = "openlm-research/open_llama_3b_v2",
    max_prompt_len: int = MAX_PROMPT_LEN,
    max_ans_len: int = MAX_ANS_LEN,
    temperature: float = 0.1,
    batch_size: int = 4,
    use_quantization: bool = False,
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

    # Create dataloader with DataCollator in inference mode (matches training tokenization)
    eval_collator = DataCollator(
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
        max_ans_len=max_ans_len,
        pad_to_multiple_of=1,
        inference=True,
        task=dataset_name
    )
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
    max_prompt_len: int = MAX_PROMPT_LEN,
    max_ans_len: int = MAX_ANS_LEN,
    temperature: float = 0.1,
    batch_size: int = 4,
    use_quantization: bool = False,
    pre_quantized: bool = False,
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
        pre_quantized: If True, models are already quantized
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
        pre_quantized=pre_quantized,
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

    # Create dataloader with DataCollator in inference mode (matches training tokenization)
    eval_collator = DataCollator(
        tokenizer=tokenizer,
        max_prompt_len=max_prompt_len,
        max_ans_len=max_ans_len,
        pad_to_multiple_of=1,
        inference=True,
        task=dataset_name  # Pass task name for dataset-specific handling
    )
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
    continual_learning: bool = True,
    include_base_model: bool = True,
    **kwargs
):
    """
    Validate all checkpoints found in the checkpoint directory.

    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        data_dir: Root directory for datasets
        output_dir: Directory to save validation results
        continual_learning: If True, validate each checkpoint on current + all previous tasks.
                          If False, validate each checkpoint only on its own training dataset.
        include_base_model: If True, also evaluate base model on all datasets (only with continual_learning=True)
        **kwargs: Additional arguments passed to validate_checkpoint
    """
    # Resolve checkpoint base directory based on USE_MERGED_MODELS
    if USE_MERGED_MODELS:
        # Convert to merged_models directory
        checkpoint_path = Path(checkpoint_base_dir)

        # Check if path already points to a models directory
        if checkpoint_path.name in ["merged_models", "quantized_models", "quantized-models"]:
            checkpoint_base = checkpoint_path
        elif "lora-continual" in checkpoint_base_dir or "continual" in checkpoint_base_dir:
            checkpoint_base = checkpoint_path.parent / "merged_models"
        else:
            checkpoint_base = checkpoint_path / "merged_models"
        logger.info(f"Using merged models directory: {checkpoint_base}")
    else:
        checkpoint_base = Path(checkpoint_base_dir)
        logger.info(f"Using LoRA checkpoints directory: {checkpoint_base}")

    # Find all checkpoint directories
    checkpoints = sorted([d for d in checkpoint_base.iterdir() if d.is_dir() and d.name.startswith('task_')])

    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_base_dir}")
        return

    logger.info(f"Found {len(checkpoints)} checkpoints")
    logger.info(f"Continual learning mode: {continual_learning}")
    logger.info(f"Include base model: {include_base_model and continual_learning}")

    all_results = []

    # Evaluate base model on all datasets (if requested and in continual learning mode)
    if continual_learning and include_base_model:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING BASE MODEL ON ALL DATASETS")
        logger.info("=" * 80)

        for dataset_name in TASK_SEQUENCE:
            try:
                logger.info(f"\nEvaluating base model on: {dataset_name}")
                evaluation_result = validate_base_model(
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    **kwargs
                )

                all_results.append({
                    'checkpoint': 'base_model',
                    'task_id': -1,
                    'dataset': dataset_name,
                    'metrics': evaluation_result
                })

            except Exception as e:
                logger.error(f"Base model validation failed on {dataset_name}: {e}")
                all_results.append({
                    'checkpoint': 'base_model',
                    'task_id': -1,
                    'dataset': dataset_name,
                    'metrics': {'error': str(e)}
                })
                continue

    # Validate each checkpoint
    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.name

        # Extract task_id and dataset name from checkpoint directory name
        # e.g., task_2_MeetingBank -> task_id=2, dataset_name=MeetingBank
        parts = checkpoint_name.split('_', 2)
        if len(parts) >= 3:
            try:
                task_id = int(parts[1])
                dataset_name = parts[2]
            except ValueError:
                logger.warning(f"Could not parse checkpoint name {checkpoint_name}, skipping")
                continue
        else:
            logger.warning(f"Could not extract info from {checkpoint_name}, skipping")
            continue

        if continual_learning:
            # Validate on current task + all previous tasks
            datasets_to_validate = TASK_SEQUENCE[:task_id + 1]

            logger.info("\n" + "=" * 80)
            logger.info(f"CHECKPOINT: {checkpoint_name}")
            logger.info(f"Validating on {len(datasets_to_validate)} datasets: {datasets_to_validate}")
            logger.info("=" * 80)

            for eval_dataset in datasets_to_validate:
                try:
                    logger.info(f"\n  Validating on: {eval_dataset}")
                    evaluation_result = validate_checkpoint(
                        checkpoint_path=str(checkpoint_path),
                        dataset_name=eval_dataset,
                        data_dir=data_dir,
                        output_dir=output_dir,
                        **kwargs
                    )

                    all_results.append({
                        'checkpoint': checkpoint_name,
                        'task_id': task_id,
                        'dataset': eval_dataset,
                        'metrics': evaluation_result
                    })

                except Exception as e:
                    logger.error(f"Validation failed for {checkpoint_path} on {eval_dataset}: {e}")
                    all_results.append({
                        'checkpoint': checkpoint_name,
                        'task_id': task_id,
                        'dataset': eval_dataset,
                        'metrics': {'error': str(e)}
                    })
                    continue
        else:
            # Original behavior: validate only on own training dataset
            try:
                logger.info("\n" + "=" * 80)
                logger.info(f"CHECKPOINT: {checkpoint_name}")
                logger.info(f"Validating on: {dataset_name}")
                logger.info("=" * 80)

                evaluation_result = validate_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    **kwargs
                )

                all_results.append({
                    'checkpoint': checkpoint_name,
                    'task_id': task_id,
                    'dataset': dataset_name,
                    'metrics': evaluation_result
                })

            except Exception as e:
                logger.error(f"Validation failed for {checkpoint_path}: {e}")
                continue

    logger.info("\n" + "=" * 80)
    logger.info("ALL VALIDATIONS COMPLETE")
    logger.info("=" * 80)

    # Save all results
    if continual_learning:
        results_path = Path(output_dir) / "continual_learning_results.json"
    else:
        results_path = Path(output_dir) / "validation_results.json"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    # Compute and save forgetting metrics if in continual learning mode
    if continual_learning and all_results:
        forgetting_metrics = compute_forgetting_metrics(all_results, output_dir)
        logger.info("\n" + "=" * 80)
        logger.info("FORGETTING METRICS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Average Forgetting: {forgetting_metrics.get('average_forgetting', 'N/A')}")
        logger.info(f"Final Average Accuracy: {forgetting_metrics.get('final_average_accuracy', 'N/A')}")
        logger.info("=" * 80)

    return all_results


def compute_forgetting_metrics(results, output_dir):
    """
    Compute continual learning metrics from validation results.

    Metrics computed:
    - Forgetting: Performance degradation on previous tasks
    - Backward Transfer: How training on new tasks affects old task performance
    - Forward Transfer: How training on previous tasks helps new task performance
    - Final Average Accuracy: Average performance on all tasks after training

    Args:
        results: List of validation results from validate_all_checkpoints
        output_dir: Directory to save metrics summary

    Returns:
        Dictionary containing computed metrics
    """
    logger.info("\nComputing forgetting metrics...")

    # Organize results by checkpoint and dataset
    # Structure: {checkpoint_name: {dataset_name: metrics}}
    results_by_checkpoint = {}

    for result in results:
        checkpoint = result['checkpoint']
        dataset = result['dataset']
        metrics = result['metrics']

        if checkpoint not in results_by_checkpoint:
            results_by_checkpoint[checkpoint] = {}

        results_by_checkpoint[checkpoint][dataset] = metrics

    # Extract performance metric (assuming 'accuracy' or 'rouge' is available)
    def get_performance(metrics):
        """Extract primary performance metric from results."""
        if 'error' in metrics:
            return None
        # Try common metric names
        for key in ['accuracy', 'rouge', 'rouge_l', 'f1', 'exact_match']:
            if key in metrics:
                return metrics[key]
        # If none found, try to get any numeric value
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                return value
        return None

    # Track performance for each dataset across checkpoints
    # Structure: {dataset_name: {checkpoint_name: performance}}
    dataset_performance = {}

    for checkpoint, datasets in results_by_checkpoint.items():
        for dataset, metrics in datasets.items():
            if dataset not in dataset_performance:
                dataset_performance[dataset] = {}

            perf = get_performance(metrics)
            if perf is not None:
                dataset_performance[dataset][checkpoint] = perf

    # Compute forgetting for each dataset
    # Forgetting = max performance on dataset - final performance on dataset
    forgetting_per_dataset = {}

    for dataset_idx, dataset_name in enumerate(TASK_SEQUENCE):
        if dataset_name not in dataset_performance:
            continue

        performances = dataset_performance[dataset_name]

        # Find the checkpoint where this task was trained (task_idx)
        task_checkpoint = f"task_{dataset_idx}_{dataset_name}"

        # Get performance on own checkpoint (if available)
        own_performance = performances.get(task_checkpoint)

        if own_performance is None:
            continue

        # Track max performance and final performance
        max_performance = own_performance
        final_performance = own_performance

        # Check performance in later checkpoints
        for task_id in range(dataset_idx + 1, len(TASK_SEQUENCE)):
            later_checkpoint = f"task_{task_id}_{TASK_SEQUENCE[task_id]}"
            if later_checkpoint in performances:
                perf = performances[later_checkpoint]
                max_performance = max(max_performance, perf)
                final_performance = perf  # Update to latest

        # Forgetting = max_performance - final_performance
        forgetting = max_performance - final_performance
        forgetting_per_dataset[dataset_name] = {
            'own_performance': own_performance,
            'max_performance': max_performance,
            'final_performance': final_performance,
            'forgetting': forgetting
        }

    # Compute average forgetting (only for tasks that have later checkpoints)
    forgetting_values = [f['forgetting'] for f in forgetting_per_dataset.values() if f['forgetting'] >= 0]
    average_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0

    # Compute final average accuracy (last checkpoint's performance on all seen tasks)
    final_checkpoint_name = None
    final_task_id = -1

    for checkpoint in results_by_checkpoint.keys():
        if checkpoint == 'base_model':
            continue
        try:
            task_id = int(checkpoint.split('_')[1])
            if task_id > final_task_id:
                final_task_id = task_id
                final_checkpoint_name = checkpoint
        except:
            continue

    final_average_accuracy = None
    if final_checkpoint_name:
        final_performances = []
        for dataset_name in TASK_SEQUENCE[:final_task_id + 1]:
            if dataset_name in results_by_checkpoint[final_checkpoint_name]:
                perf = get_performance(results_by_checkpoint[final_checkpoint_name][dataset_name])
                if perf is not None:
                    final_performances.append(perf)

        if final_performances:
            final_average_accuracy = sum(final_performances) / len(final_performances)

    # Prepare summary
    metrics_summary = {
        'average_forgetting': average_forgetting,
        'final_average_accuracy': final_average_accuracy,
        'forgetting_per_dataset': forgetting_per_dataset,
        'dataset_performance_matrix': dataset_performance
    }

    # Save detailed metrics
    metrics_path = Path(output_dir) / "continual_learning_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Forgetting metrics saved to: {metrics_path}")

    # Print per-dataset forgetting
    logger.info("\nPer-dataset forgetting:")
    for dataset, metrics in forgetting_per_dataset.items():
        logger.info(f"  {dataset}:")
        logger.info(f"    Own performance: {metrics['own_performance']:.4f}")
        logger.info(f"    Max performance: {metrics['max_performance']:.4f}")
        logger.info(f"    Final performance: {metrics['final_performance']:.4f}")
        logger.info(f"    Forgetting: {metrics['forgetting']:.4f}")

    return metrics_summary


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
    # Resolve checkpoint base directory based on USE_MERGED_MODELS
    if USE_MERGED_MODELS:
        # Convert to merged_models directory
        checkpoint_path = Path(checkpoint_base_dir)

        # Check if path already points to a models directory
        if checkpoint_path.name in ["merged_models", "quantized_models", "quantized-models"]:
            checkpoint_base = checkpoint_path
        elif "lora-continual" in checkpoint_base_dir or "continual" in checkpoint_base_dir:
            checkpoint_base = checkpoint_path.parent / "merged_models"
        else:
            checkpoint_base = checkpoint_path / "merged_models"
        logger.info(f"Using merged models directory: {checkpoint_base}")
    else:
        checkpoint_base = Path(checkpoint_base_dir)
        logger.info(f"Using LoRA checkpoints directory: {checkpoint_base}")

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
  # CONTINUAL LEARNING VALIDATION (RECOMMENDED)
  # Validate all checkpoints with continual learning evaluation
  # Each checkpoint is tested on its current task + all previous tasks
  # Also evaluates base model on all datasets and computes forgetting metrics
  python validate.py --validate-all ./lora-continual
        """
    )

    # Checkpoint selection (mutually exclusive)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Path to specific checkpoint directory to validate"
    )
    checkpoint_group.add_argument(
        "--validate-all",
        type=str,
        metavar="CHECKPOINT_DIR",
        help="Validate all checkpoints in the specified directory (with continual learning mode enabled by default)"
    )
    checkpoint_group.add_argument(
        "--eval-all-on-dataset",
        type=str,
        metavar="DATASET",
        help="Evaluate all checkpoints on a single specified dataset (useful for measuring forgetting)"
    )
    checkpoint_group.add_argument(
        "--eval-base-model",
        type=str,
        metavar="DATASET",
        help="Evaluate only the base model (no LoRA) on a specified dataset"
    )

    # Validation mode options
    parser.add_argument(
        "--no-continual-learning",
        action="store_true",
        help="Disable continual learning mode: validate each checkpoint only on its own training dataset (default: enabled)"
    )
    parser.add_argument(
        "--no-base-model",
        action="store_true",
        help="Exclude base model evaluation (only applies to --validate-all and --eval-all-on-dataset, default: included)"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="./lora-continual",
        help="Base directory containing checkpoints for --eval-all-on-dataset mode (default: ./lora-continual)"
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
        help="Base directory for validation results. Subdirectories with checkpoint name and timestamp will be created automatically (default: ./validation-results)"
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
        help="Enable 4-bit quantization (default: disabled)"
    )
    parser.add_argument(
        "--pre-quantized",
        action="store_true",
        help="Models are already quantized (don't apply quantization config during loading)"
    )

    # Generation settings
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
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        choices=DEFAULT_TASK_CONFIGS.keys(),
        help="Can be passed multiple times.",
    )

    args = parser.parse_args()
    global TASK_SEQUENCE
    TASK_SEQUENCE=args.task
    # Determine boolean settings
    continual_learning = not args.no_continual_learning  # Enabled by default
    include_base_model = not args.no_base_model          # Enabled by default
    use_quantization = args.use_quantization             # Disabled by default

    # Create timestamped output directory based on validation mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir

    if args.validate_all:
        output_dir = Path(base_output_dir) / f"all_checkpoints_{timestamp}"
    elif args.eval_all_on_dataset:
        output_dir = Path(base_output_dir) / f"{args.eval_all_on_dataset}_{timestamp}"
    elif args.eval_base_model:
        output_dir = Path(base_output_dir) / f"base_model_{args.eval_base_model}_{timestamp}"
    else:
        checkpoint_name = Path(args.checkpoint_dir).name
        output_dir = Path(base_output_dir) / f"{checkpoint_name}_{timestamp}"

    logger.info(f"Output directory: {output_dir}")

    common_kwargs = {
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "base_model_name": args.base_model,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "use_quantization": use_quantization,
        "pre_quantized": args.pre_quantized,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Run validation
    if args.validate_all:
        validate_all_checkpoints(
            checkpoint_base_dir=args.validate_all,
            continual_learning=continual_learning,
            include_base_model=include_base_model,
            **common_kwargs
        )
    elif args.eval_all_on_dataset:
        validate_all_on_single_dataset(
            dataset_name=args.eval_all_on_dataset,
            checkpoint_base_dir=args.checkpoint_base_dir,
            include_base_model=include_base_model,
            **common_kwargs
        )
    elif args.eval_base_model:
        validate_base_model(
            dataset_name=args.eval_base_model,
            **common_kwargs
        )
    else:
        # Single checkpoint validation
        checkpoint_path_input = Path(args.checkpoint_dir)
        checkpoint_name = checkpoint_path_input.name

        if USE_MERGED_MODELS:
            # Convert path to merged_models directory
            checkpoint_dir_str = str(checkpoint_path_input)
            if "lora-continual" in checkpoint_dir_str or "continual" in checkpoint_dir_str:
                # Replace continual with merged_models
                checkpoint_path = checkpoint_path_input.parent.parent / "merged_models" / checkpoint_name
                logger.info(f"Resolved to merged model path: {checkpoint_path}")
            else:
                # Already pointing to correct location or use as-is
                checkpoint_path = checkpoint_path_input
        else:
            checkpoint_path = checkpoint_path_input

        # Extract task_id from checkpoint directory name
        # e.g., task_2_MeetingBank -> task_id=2, dataset_name=MeetingBank
        parts = checkpoint_name.split('_', 2)
        if len(parts) >= 3:
            try:
                task_id = int(parts[1])
                checkpoint_dataset_name = parts[2]
            except ValueError:
                parser.error(f"Could not parse checkpoint name {checkpoint_name}")
        else:
            parser.error(f"Could not extract info from {checkpoint_name}, expected format: task_N_DatasetName")

        # Determine datasets to validate on
        if args.dataset:
            # User specified a single dataset, validate only on that
            datasets_to_validate = [args.dataset]
            logger.info(f"Validating on user-specified dataset: {args.dataset}")
        elif continual_learning:
            # Continual learning mode: validate on current + all previous tasks
            datasets_to_validate = TASK_SEQUENCE[:task_id + 1]
            logger.info(f"Continual learning mode: validating on {len(datasets_to_validate)} datasets: {datasets_to_validate}")
        else:
            # Non-continual learning mode: validate only on checkpoint's own dataset
            datasets_to_validate = [checkpoint_dataset_name]
            logger.info(f"Validating on checkpoint's training dataset: {checkpoint_dataset_name}")

        logger.info("=" * 80)
        logger.info(f"CHECKPOINT: {checkpoint_name}")
        logger.info(f"Datasets to validate: {datasets_to_validate}")
        logger.info("=" * 80)

        # Validate on each dataset
        all_results = []
        for dataset_name in datasets_to_validate:
            try:
                logger.info(f"\nValidating on: {dataset_name}")
                evaluation_result = validate_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    dataset_name=dataset_name,
                    **common_kwargs
                )

                all_results.append({
                    'checkpoint': checkpoint_name,
                    'task_id': task_id,
                    'dataset': dataset_name,
                    'metrics': evaluation_result
                })

            except Exception as e:
                logger.error(f"Validation failed on {dataset_name}: {e}")
                all_results.append({
                    'checkpoint': checkpoint_name,
                    'task_id': task_id,
                    'dataset': dataset_name,
                    'metrics': {'error': str(e)}
                })
                continue

        # Save all results
        results_path = Path(output_dir) / "validation_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info(f"Results saved to: {results_path}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
