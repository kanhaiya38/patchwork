"""
LoRA Continual Learning Training with Auto-Resume

This script handles continual learning with automatic checkpoint detection and resume.
It automatically detects the last successful checkpoint and resumes training from there.

Usage:
    # Auto-resume from last checkpoint (default)
    python training.py

    # Start fresh (ignore existing checkpoints)
    python training.py --no-resume

    # Custom output directory
    python training.py --output-base-dir ./my-experiment --batch-size 16
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from datasets import load_dataset, load_from_disk
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a training task in the continual learning sequence."""

    dataset_name: str
    num_epochs: int

    def __str__(self) -> str:
        return f"{self.dataset_name} ({self.num_epochs} epochs)"


class ContinualLearningTrainer:
    """Handles continual learning with automatic checkpoint detection and resume."""

    def __init__(
        self,
        base_model_name: str = "openlm-research/open_llama_3b_v2",
        output_base_dir: str = "./experiments",
        data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_5000",
        max_prompt_len: int = 1024,
        max_ans_len: int = 512,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
    ):
        """
        Initialize the continual learning trainer.

        Args:
            base_model_name: HuggingFace model identifier
            output_base_dir: Base directory for all outputs (will create 'continual' and 'checkpoints' subdirs)
            data_dir: Root directory for datasets
            max_prompt_len: Maximum prompt length in tokens
            max_ans_len: Maximum answer length in tokens
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        self.base_model_name = base_model_name
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / "continual"  # Task checkpoints
        self.training_output_dir = (
            self.output_base_dir / "checkpoints"
        )  # Training artifacts
        self.data_dir = Path(data_dir)

        # Create cache dir using both prompt and answer lengths for uniqueness
        data_dir_name = self.data_dir.name
        cache_suffix = f"prompt{max_prompt_len}_ans{max_ans_len}"
        self.cache_dir = Path(".cache") / "tokenized_datasets" / data_dir_name / cache_suffix

        self.max_prompt_len = max_prompt_len
        self.max_ans_len = max_ans_len
        self.max_length = max_prompt_len + max_ans_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def find_last_checkpoint(
        self, tasks: List[Task]
    ) -> Tuple[Optional[int], Optional[Path]]:
        """
        Find the last successful checkpoint.

        Args:
            tasks: List of Task objects

        Returns:
            Tuple of (last_completed_task_id, checkpoint_path) or (None, None) if no checkpoint found
        """
        logger.info("Searching for existing checkpoints...")

        for task_id in range(len(tasks) - 1, -1, -1):
            task = tasks[task_id]
            checkpoint_path = self.output_dir / f"task_{task_id}_{task.dataset_name}"

            if (
                checkpoint_path.exists()
                and (checkpoint_path / "adapter_config.json").exists()
            ):
                logger.info(f"Found checkpoint: {checkpoint_path}")
                return task_id, checkpoint_path

        logger.info("No existing checkpoints found")
        return None, None

    def setup_model_and_tokenizer(
        self, checkpoint_path: Optional[Path] = None, use_quantization: bool = True
    ) -> None:
        """
        Load model and tokenizer, optionally from a checkpoint.

        Args:
            checkpoint_path: Path to LoRA checkpoint to resume from
            use_quantization: Whether to use 4-bit quantization
        """
        logger.info("Setting up model and tokenizer...")

        # Configure quantization
        bnb_config = None
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        # Load base model
        logger.info(f"Loading base model from {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load from checkpoint or prepare new model
        if checkpoint_path:
            logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            # Ensure model is in training mode and gradients are enabled
            model.train()
            # Enable gradient checkpointing if using quantization
            # Enable gradient checkpointing and input gradients
            if not use_quantization:
                # For non-quantized models, explicitly enable gradient checkpointing
                base_model.gradient_checkpointing_enable()
            # Always enable input gradients for gradient checkpointing to work
            model.enable_input_require_grads()
        else:
            logger.info("Preparing new LoRA model from scratch...")
            if use_quantization:
                base_model = prepare_model_for_kbit_training(base_model)
            else:
                # Enable gradient checkpointing for memory efficiency
                base_model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(base_model, lora_config)

        model.print_trainable_parameters()
        self.model = model
        logger.info("Model and tokenizer setup complete")

    def get_tokenized_dataset(self, dataset_name: str):
        """
        Load and tokenize dataset with automatic caching.
        Cache is stored in .cache/tokenized_datasets/{data_dir_name}/ directory.

        Args:
            dataset_name: Name of the dataset subdirectory

        Returns:
            Tokenized dataset ready for training
        """
        assert (
            self.tokenizer is not None
        ), "Tokenizer not initialized. Call setup_model_and_tokenizer first."

        # Check for cached tokenized dataset
        cache_path = self.cache_dir / dataset_name

        if cache_path.exists():
            logger.info(f"Loading cached tokenized dataset from: {cache_path}")
            try:
                tokenized_dataset = load_from_disk(str(cache_path))
                logger.info(
                    f"✓ Cached dataset loaded (skipped formatting & tokenization)"
                )
                return tokenized_dataset
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Re-processing...")

        logger.info(f"Loading dataset: {dataset_name}")

        dataset = load_dataset(
            "json",
            data_files={
                "train": str(self.data_dir / dataset_name / "train.json"),
                "test": str(self.data_dir / dataset_name / "test.json"),
                "eval": str(self.data_dir / dataset_name / "eval.json"),
            },
        )

        def format_instruction(examples):
            # Don't add EOS here - will be added explicitly during tokenization
            texts = [
                prompt + answer
                for prompt, answer in zip(examples["prompt"], examples["answer"])
            ]
            # Keep 'answer' field for dynamic masking in collator
            return {"text": texts, "answer": examples["answer"]}

        logger.info(f"Formatting dataset: {dataset_name}")
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        logger.info(f"Train size: {len(dataset['train'])}")
        MAP_BATCH_SIZE = 10000  # Batch size for dataset.map() operations
        formatted_dataset = dataset.map(
            format_instruction,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            remove_columns=["prompt"],  # Keep 'answer' field
        )
        logger.info(f"Formatting completed successfully")

        # Capture tokenizer and max_length to avoid pickling self
        tokenizer = self.tokenizer
        max_length = self.max_length
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        def tokenize_function(examples):
            # Tokenize without automatic special tokens (TRACE-style)
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length - 2,  # Reserve space for BOS + EOS
                add_special_tokens=False,   # Explicit control over special tokens
                padding=False
            )

            # Manually add BOS and EOS tokens to each sequence
            result_ids = []
            result_mask = []
            for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
                # Add EOS token if space available
                if len(ids) < max_length - 1:
                    ids.append(eos_token_id)
                    mask.append(1)
                # Prepend BOS token if space available
                if len(ids) < max_length:
                    ids = [bos_token_id] + ids
                    mask = [1] + mask
                result_ids.append(ids)
                result_mask.append(mask)

            return {
                "input_ids": result_ids,
                "attention_mask": result_mask,
                "answer": examples["answer"],
            }

        logger.info(f"Tokenizing dataset: {dataset_name}")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            remove_columns=["text"],  # Keep 'answer' field for collator
        )
        logger.info(f"Tokenization completed successfully")

        # Save to cache for future use
        logger.info(f"Saving tokenized dataset to cache: {cache_path}")
        tokenized_dataset.save_to_disk(str(cache_path))

        logger.info(f"Dataset {dataset_name} prepared successfully")
        return tokenized_dataset

    def data_collator_with_prompt_masking(self, features):
        """
        Custom data collator that masks prompt tokens in labels.
        Dynamically calculates answer length for each example to work with all TRACE datasets.

        Works for any answer length:
        - Short answers (FOMC: "A" = 1 token)
        - Long answers (MeetingBank: summaries = 200+ tokens)
        """
        # Determine max length for padding
        max_length = max(len(feature["input_ids"]) for feature in features)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for idx, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]

            # Check if 'answer' key exists
            if "answer" not in feature:
                logger.error(
                    f"Feature {idx} missing 'answer' key. Available keys: {list(feature.keys())}"
                )
                raise KeyError(
                    "'answer' field is missing from features. Check dataset processing."
                )

            answer_text = feature["answer"]

            # Dynamically calculate answer length by tokenizing it
            # This handles any answer length (from 1 token to 500+ tokens)
            answer_with_eos = answer_text + self.tokenizer.eos_token
            tokenized_answer = self.tokenizer(
                answer_with_eos,
                add_special_tokens=False,  # Don't add BOS, we only want answer + EOS
                truncation=False,
            )
            answer_length = len(tokenized_answer["input_ids"])

            # Create labels: mask prompt tokens, keep answer tokens
            # -100 is the ignore index for CrossEntropyLoss
            prompt_length = len(input_ids) - answer_length
            if prompt_length < 0:
                # Safety check: if answer is longer than full sequence, something is wrong
                logger.warning(
                    f"Example {idx}: answer_length ({answer_length}) > total_length ({len(input_ids)})"
                )
                prompt_length = 0

            labels = [-100] * prompt_length + input_ids[-answer_length:]

            # Pad sequences
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch

    def train_model(self, tokenized_dataset, num_epochs: int) -> None:
        """
        Train the model on a dataset.

        Args:
            tokenized_dataset: Preprocessed dataset
            num_epochs: Number of training epochs
        """
        assert (
            self.model is not None
        ), "Model not initialized. Call setup_model_and_tokenizer first."
        assert (
            self.tokenizer is not None
        ), "Tokenizer not initialized. Call setup_model_and_tokenizer first."

        logger.info(f"Starting training for {num_epochs} epochs...")

        training_args = TrainingArguments(
            output_dir=str(self.training_output_dir),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epochs,
            learning_rate=self.learning_rate,
            bf16=True,  # Use bfloat16 instead of fp16 for better stability with quantized models
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
            warmup_steps=50,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            remove_unused_columns=False,  # Keep 'answer' field for data collator
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=self.data_collator_with_prompt_masking,
        )

        trainer.train()
        logger.info("Training completed")

    def save_checkpoint(self, task_id: int, dataset_name: str) -> Path:
        """
        Save model checkpoint.

        Args:
            task_id: Task ID number
            dataset_name: Name of the dataset

        Returns:
            Path to saved checkpoint
        """
        assert self.model is not None, "Model not initialized."
        assert self.tokenizer is not None, "Tokenizer not initialized."

        checkpoint_path = self.output_dir / f"task_{task_id}_{dataset_name}"
        logger.info(f"Saving checkpoint to {checkpoint_path}")

        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))

        logger.info(f"Checkpoint saved successfully")
        return checkpoint_path

    def run_continual_learning(
        self,
        tasks: List[Task],
        auto_resume: bool = True,
        use_quantization: bool = True,
    ) -> None:
        """
        Run continual learning on a sequence of tasks with auto-resume.

        Args:
            tasks: List of Task objects
            auto_resume: Whether to automatically resume from last checkpoint
            use_quantization: Whether to use 4-bit quantization
        """
        # Find last checkpoint if auto-resume is enabled
        start_task_id = 0
        checkpoint_path = None

        if auto_resume:
            last_task_id, checkpoint_path = self.find_last_checkpoint(tasks)

            if last_task_id is not None:
                start_task_id = last_task_id + 1
                logger.info("=" * 80)
                logger.info(f"RESUMING FROM CHECKPOINT")
                logger.info(
                    f"Last completed task: {last_task_id} ({tasks[last_task_id]})"
                )
                logger.info(f"Resuming from task: {start_task_id}")
                logger.info("=" * 80)

                if start_task_id >= len(tasks):
                    logger.info("All tasks already completed!")
                    return
            else:
                logger.info("=" * 80)
                logger.info("STARTING FRESH (no checkpoints found)")
                logger.info("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("STARTING FRESH (auto-resume disabled)")
            logger.info("=" * 80)

        # Setup model
        self.setup_model_and_tokenizer(
            checkpoint_path=checkpoint_path, use_quantization=use_quantization
        )

        # Train on remaining tasks
        for task_id in range(start_task_id, len(tasks)):
            task = tasks[task_id]

            logger.info("=" * 80)
            logger.info(f"TASK {task_id}/{len(tasks)-1}: {task}")
            logger.info("=" * 80)

            try:
                # Prepare dataset
                tokenized_dataset = self.get_tokenized_dataset(task.dataset_name)

                # Train
                self.train_model(tokenized_dataset, num_epochs=task.num_epochs)

                # Save checkpoint
                self.save_checkpoint(task_id, task.dataset_name)

                logger.info(f"✓ Task {task_id} completed successfully")

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"✗ Task {task_id} FAILED with error:")
                logger.error(f"{type(e).__name__}: {e}")
                logger.error("=" * 80)

                if task_id > 0:
                    logger.info(
                        f"Last successful checkpoint: task_{task_id-1}_{tasks[task_id-1].dataset_name}"
                    )
                    logger.info(f"To resume, simply run: python training.py")
                else:
                    logger.info(
                        "Training failed on first task. No checkpoint available."
                    )

                raise

        logger.info("=" * 80)
        logger.info("🎉 ALL TASKS COMPLETED SUCCESSFULLY! 🎉")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LoRA Continual Learning with Auto-Resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-resume from last checkpoint (default)
  python training.py

  # Start fresh, ignoring existing checkpoints
  python training.py --no-resume

  # Custom output directory
  python training.py --output-base-dir ./my-experiment

  # Custom configuration
  python training.py --output-base-dir ./my-experiment --batch-size 16
        """,
    )

    # Training control
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch, ignoring existing checkpoints",
    )

    # Paths
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="./experiments",
        help="Base directory for outputs (creates 'continual' and 'checkpoints' subdirs, default: ./experiments)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32, increase to 128+ for H200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=1024,
        help="Maximum prompt length in tokens (default: 1024)",
    )
    parser.add_argument(
        "--max-ans-len",
        type=int,
        default=512,
        help="Maximum answer length in tokens (default: 512)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable 4-bit quantization (useful for smaller GPUs, slower on H200)",
    )

    args = parser.parse_args()

    # Define all tasks
    tasks = [
        Task(dataset_name='C-STANCE', num_epochs=5),
        Task(dataset_name="FOMC", num_epochs=3),
        # Task(dataset_name='MeetingBank', num_epochs=7),
        # Task(dataset_name='Py150', num_epochs=5),
        # Task(dataset_name='ScienceQA', num_epochs=3),
        # Task(dataset_name='NumGLUE-cm', num_epochs=5),
        # Task(dataset_name='NumGLUE-ds', num_epochs=5),
        # Task(dataset_name='20Minuten', num_epochs=7),
    ]

    logger.info("=" * 80)
    logger.info("CONTINUAL LEARNING TRAINING")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Tasks: {[task.dataset_name for task in tasks]}")
    logger.info(f"Output directory: {args.output_base_dir}")
    logger.info(f"  - Task checkpoints: {args.output_base_dir}/continual")
    logger.info(f"  - Training artifacts: {args.output_base_dir}/checkpoints")
    logger.info(f"Auto-resume: {not args.no_resume}")
    logger.info(
        f"Quantization: {'enabled (4-bit)' if args.quantize else 'disabled (full precision)'}"
    )
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max prompt length: {args.max_prompt_len}")
    logger.info(f"Max answer length: {args.max_ans_len}")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = ContinualLearningTrainer(
        output_base_dir=args.output_base_dir,
        data_dir="TRACE-Benchmark/LLM-CL-Benchmark_5000",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Run continual learning
    trainer.run_continual_learning(
        tasks=tasks,
        auto_resume=not args.no_resume,
        use_quantization=args.quantize,
    )


if __name__ == "__main__":
    main()
