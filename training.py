"""
LoRA Continual Learning Training with Auto-Resume and Experience Replay

This script handles continual learning with:
- Automatic checkpoint detection and resume
- Experience replay to reduce catastrophic forgetting (additive mode)
- Configurable replay buffer size

Replay Strategy:
    Additive approach - All current task samples + replay samples from previous tasks
    Example: 5000 (current) + 500 (replay) = 5500 total samples per task

Usage:
    # Auto-resume from last checkpoint with experience replay (default)
    python training.py

    # Start fresh (ignore existing checkpoints)
    python training.py --no-resume

    # Custom replay configuration (store more samples per task)
    python training.py --replay-samples-per-task 1000

    # Custom output directory
    python training.py --output-base-dir ./my-experiment --batch-size 16
"""

import argparse
import logging
import sys
import random
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
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
    batch_size: Optional[int] = None  # Optional task-specific batch size

    def __str__(self) -> str:
        batch_info = f", batch_size={self.batch_size}" if self.batch_size else ""
        return f"{self.dataset_name} ({self.num_epochs} epochs{batch_info})"


class ExperienceReplayBuffer:
    """
    Stores representative samples from previous tasks for experience replay.
    Reduces catastrophic forgetting by rehearsing old examples during new task training.
    """

    def __init__(self, max_samples_per_task: int = 500, selection_strategy: str = "random"):
        """
        Initialize the experience replay buffer.

        Args:
            max_samples_per_task: Maximum number of samples to store per task
            selection_strategy: Strategy for selecting samples ('random', 'diverse', 'balanced')
        """
        self.max_samples_per_task = max_samples_per_task
        self.selection_strategy = selection_strategy
        self.buffer: Dict[int, Dataset] = {}  # task_id -> Dataset of stored samples
        self.task_info: Dict[int, Dict] = {}  # task_id -> metadata

    def add_task_samples(self, task_id: int, dataset: Dataset, task_name: str) -> None:
        """
        Store representative samples from a completed task.

        Args:
            task_id: ID of the completed task
            dataset: Full training dataset from the task
            task_name: Name of the task for logging
        """
        num_samples = min(self.max_samples_per_task, len(dataset))

        logger.info(f"Adding {num_samples} samples from task {task_id} ({task_name}) to replay buffer")

        if self.selection_strategy == "random":
            # Randomly sample from the dataset
            indices = random.sample(range(len(dataset)), num_samples)
            selected_samples = dataset.select(indices)
        elif self.selection_strategy == "first":
            # Take first N samples (deterministic, useful for debugging)
            selected_samples = dataset.select(range(num_samples))
        else:
            # Default to random
            indices = random.sample(range(len(dataset)), num_samples)
            selected_samples = dataset.select(indices)

        self.buffer[task_id] = selected_samples
        self.task_info[task_id] = {
            "task_name": task_name,
            "num_samples": num_samples,
            "total_dataset_size": len(dataset),
        }

        logger.info(f"✓ Replay buffer now contains {len(self.buffer)} tasks, "
                   f"total samples: {sum(len(ds) for ds in self.buffer.values())}")

    def get_replay_dataset(self, current_task_id: int) -> Optional[Dataset]:
        """
        Get combined dataset from all previous tasks for replay.

        Args:
            current_task_id: ID of the current task being trained

        Returns:
            Combined dataset from all previous tasks, or None if no previous tasks
        """
        if current_task_id == 0 or len(self.buffer) == 0:
            return None

        # Collect all samples from previous tasks
        replay_datasets = []
        for task_id in range(current_task_id):
            if task_id in self.buffer:
                replay_datasets.append(self.buffer[task_id])

        if not replay_datasets:
            return None

        # Concatenate all replay datasets
        combined_replay = concatenate_datasets(replay_datasets)

        logger.info(f"Replay dataset contains {len(combined_replay)} samples "
                   f"from {len(replay_datasets)} previous task(s)")

        return combined_replay

    def save(self, path: Path) -> None:
        """
        Save the replay buffer to disk for resume functionality.

        Args:
            path: Directory path to save the buffer
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save buffer datasets
        for task_id, dataset in self.buffer.items():
            dataset_path = path / f"task_{task_id}_replay"
            dataset.save_to_disk(str(dataset_path))

        # Save metadata
        metadata = {
            "max_samples_per_task": self.max_samples_per_task,
            "selection_strategy": self.selection_strategy,
            "task_info": self.task_info,
        }
        metadata_path = path / "replay_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"✓ Replay buffer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> Optional["ExperienceReplayBuffer"]:
        """
        Load the replay buffer from disk.

        Args:
            path: Directory path containing the saved buffer

        Returns:
            Loaded ExperienceReplayBuffer or None if path doesn't exist
        """
        if not path.exists():
            return None

        metadata_path = path / "replay_metadata.pkl"
        if not metadata_path.exists():
            logger.warning(f"No replay metadata found at {path}")
            return None

        try:
            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Create buffer instance
            buffer = cls(
                max_samples_per_task=metadata["max_samples_per_task"],
                selection_strategy=metadata["selection_strategy"],
            )
            buffer.task_info = metadata["task_info"]

            # Load datasets
            for task_id in buffer.task_info.keys():
                dataset_path = path / f"task_{task_id}_replay"
                if dataset_path.exists():
                    buffer.buffer[task_id] = load_from_disk(str(dataset_path))
                else:
                    logger.warning(f"Missing replay dataset for task {task_id}")

            total_samples = sum(len(ds) for ds in buffer.buffer.values())
            logger.info(f"✓ Replay buffer loaded from {path} "
                       f"({len(buffer.buffer)} tasks, {total_samples} samples)")

            return buffer

        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get statistics about the replay buffer."""
        return {
            "num_tasks": len(self.buffer),
            "total_samples": sum(len(ds) for ds in self.buffer.values()),
            "samples_per_task": {
                task_id: len(ds) for task_id, ds in self.buffer.items()
            },
            "task_names": {
                task_id: info["task_name"] for task_id, info in self.task_info.items()
            },
        }


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
        use_experience_replay: bool = True,
        replay_samples_per_task: int = 500,
        replay_mix_ratio: float = 0.3,
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
            use_experience_replay: Enable experience replay for reducing forgetting
            replay_samples_per_task: Number of samples to store per task in replay buffer
            replay_mix_ratio: (Deprecated - kept for compatibility) All replay samples are now added on top
        """
        self.base_model_name = base_model_name
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / "continual"  # Task checkpoints
        self.training_output_dir = (
            self.output_base_dir / "checkpoints"
        )  # Training artifacts
        self.replay_buffer_dir = self.output_base_dir / "replay_buffer"  # Replay buffer storage
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

        # Experience replay configuration
        self.use_experience_replay = use_experience_replay
        self.replay_mix_ratio = replay_mix_ratio
        self.replay_buffer: Optional[ExperienceReplayBuffer] = None
        if use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(
                max_samples_per_task=replay_samples_per_task,
                selection_strategy="random"
            )

        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer_dir.mkdir(parents=True, exist_ok=True)
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

    def train_model(
        self,
        tokenized_dataset,
        num_epochs: int,
        batch_size: Optional[int] = None,
        replay_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Train the model on a dataset, optionally adding replay samples.

        Uses additive replay: Keeps ALL current task samples and adds replay samples on top.
        Example: 5000 current + 500 replay = 5500 total samples

        Args:
            tokenized_dataset: Preprocessed dataset for current task
            num_epochs: Number of training epochs
            batch_size: Optional task-specific batch size (uses default if not provided)
            replay_dataset: Optional dataset of replay samples from previous tasks
        """
        assert (
            self.model is not None
        ), "Model not initialized. Call setup_model_and_tokenizer first."
        assert (
            self.tokenizer is not None
        ), "Tokenizer not initialized. Call setup_model_and_tokenizer first."

        # Use task-specific batch size if provided, otherwise use default
        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        # Prepare training dataset with optional replay addition
        train_dataset = tokenized_dataset["train"]

        if replay_dataset is not None and len(replay_dataset) > 0:
            logger.info("=" * 60)
            logger.info("EXPERIENCE REPLAY ENABLED (Additive Mode)")
            logger.info(f"Current task samples: {len(train_dataset)}")
            logger.info(f"Replay buffer samples: {len(replay_dataset)}")

            # Additive approach: Keep ALL current task samples + add replay samples
            # This ensures full learning on new task while preventing forgetting
            current_task_size = len(train_dataset)
            replay_size = len(replay_dataset)

            logger.info(f"Mixed dataset composition:")
            logger.info(f"  - Current task: {current_task_size} samples (100% of task data)")
            logger.info(f"  - Replay buffer: {replay_size} samples (added on top)")

            # Concatenate full datasets and shuffle for better mixing
            train_dataset = concatenate_datasets([train_dataset, replay_dataset]).shuffle(seed=42)

            logger.info(f"Total training dataset size: {len(train_dataset)} samples")
            logger.info(f"  ({current_task_size} new + {replay_size} replay)")
            logger.info("=" * 60)
        else:
            logger.info(f"Training on current task only ({len(train_dataset)} samples)")

        logger.info(f"Starting training for {num_epochs} epochs with batch size {effective_batch_size}...")

        training_args = TrainingArguments(
            output_dir=str(self.training_output_dir),
            per_device_train_batch_size=effective_batch_size,
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
            train_dataset=train_dataset,
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

                # Load replay buffer if resuming and replay is enabled
                if self.use_experience_replay and self.replay_buffer is not None:
                    loaded_buffer = ExperienceReplayBuffer.load(self.replay_buffer_dir)
                    if loaded_buffer is not None:
                        self.replay_buffer = loaded_buffer
                        stats = self.replay_buffer.get_stats()
                        logger.info(f"Replay buffer loaded: {stats['num_tasks']} tasks, "
                                  f"{stats['total_samples']} samples")
                    else:
                        logger.warning("Could not load replay buffer, starting fresh buffer")

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

                # Get replay dataset from previous tasks if replay is enabled
                replay_dataset = None
                if self.use_experience_replay and self.replay_buffer is not None:
                    replay_dataset = self.replay_buffer.get_replay_dataset(task_id)

                # Train (use task-specific batch size if provided)
                self.train_model(
                    tokenized_dataset,
                    num_epochs=task.num_epochs,
                    batch_size=task.batch_size,
                    replay_dataset=replay_dataset
                )

                # Save checkpoint
                self.save_checkpoint(task_id, task.dataset_name)

                # Add samples from current task to replay buffer for future tasks
                if self.use_experience_replay and self.replay_buffer is not None:
                    self.replay_buffer.add_task_samples(
                        task_id=task_id,
                        dataset=tokenized_dataset["train"],
                        task_name=task.dataset_name
                    )
                    # Save replay buffer after each task
                    self.replay_buffer.save(self.replay_buffer_dir)

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
        description="LoRA Continual Learning with Auto-Resume and Experience Replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-resume from last checkpoint with experience replay (default)
  python training.py

  # Start fresh, ignoring existing checkpoints
  python training.py --no-resume

  # Custom output directory
  python training.py --output-base-dir ./my-experiment

  # Disable experience replay (not recommended)
  python training.py --no-replay

  # Custom replay configuration (store more samples per task)
  python training.py --replay-samples-per-task 1000

  # Full custom configuration
  python training.py --output-base-dir ./my-experiment --batch-size 16 --learning-rate 3e-5 --replay-samples-per-task 800
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

    # Experience replay arguments
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Disable experience replay (not recommended, increases forgetting)",
    )
    parser.add_argument(
        "--replay-samples-per-task",
        type=int,
        default=500,
        help="Number of samples to store per task in replay buffer (default: 500)",
    )
    parser.add_argument(
        "--replay-mix-ratio",
        type=float,
        default=0.3,
        help="(Deprecated) No longer used. Replay samples are added on top of full task data.",
    )

    args = parser.parse_args()

    # Define all tasks
    # Note: MeetingBank uses batch_size=8 due to long sequences (200+ tokens)
    # to avoid OOM errors. Other datasets can use the default batch size.
    tasks = [
        Task(dataset_name='C-STANCE', num_epochs=5),
        Task(dataset_name="FOMC", num_epochs=3),
        Task(dataset_name='MeetingBank', num_epochs=7, batch_size=32),
        Task(dataset_name='Py150', num_epochs=5, batch_size=32),
        Task(dataset_name='ScienceQA', num_epochs=3, batch_size=64),
        Task(dataset_name='NumGLUE-cm', num_epochs=5),
        Task(dataset_name='NumGLUE-ds', num_epochs=5),
        Task(dataset_name='20Minuten', num_epochs=7, batch_size=32),
    ]

    logger.info("=" * 80)
    logger.info("CONTINUAL LEARNING TRAINING")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Tasks: {[task.dataset_name for task in tasks]}")
    logger.info(f"Output directory: {args.output_base_dir}")
    logger.info(f"  - Task checkpoints: {args.output_base_dir}/continual")
    logger.info(f"  - Training artifacts: {args.output_base_dir}/checkpoints")
    logger.info(f"  - Replay buffer: {args.output_base_dir}/replay_buffer")
    logger.info(f"Auto-resume: {not args.no_resume}")
    logger.info(
        f"Quantization: {'enabled (4-bit)' if args.quantize else 'disabled (full precision)'}"
    )
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Max prompt length: {args.max_prompt_len}")
    logger.info(f"Max answer length: {args.max_ans_len}")
    logger.info("--- Experience Replay Configuration ---")
    if not args.no_replay:
        logger.info(f"Experience Replay: ENABLED (Additive Mode)")
        logger.info(f"  - Samples stored per task: {args.replay_samples_per_task}")
        logger.info(f"  - Mode: All replay samples added on top of full task data")
        logger.info(f"  - Growth: Task dataset increases by ~{args.replay_samples_per_task} samples per task")
    else:
        logger.info(f"Experience Replay: DISABLED")
        logger.info(f"  ⚠️  Warning: Catastrophic forgetting will be higher without replay!")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = ContinualLearningTrainer(
        output_base_dir=args.output_base_dir,
        data_dir="TRACE-Benchmark/LLM-CL-Benchmark_5000",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_experience_replay=not args.no_replay,
        replay_samples_per_task=args.replay_samples_per_task,
        replay_mix_ratio=args.replay_mix_ratio,
    )

    # Run continual learning
    trainer.run_continual_learning(
        tasks=tasks,
        auto_resume=not args.no_resume,
        use_quantization=args.quantize,
    )


if __name__ == "__main__":
    main()
