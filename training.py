"""
LoRA Continual Learning Training with Auto-Resume and Experience Replay

This script handles continual learning with:
- Automatic checkpoint detection and resume (including intermediate checkpoints)
- Experience replay to reduce catastrophic forgetting (additive mode)
- Configurable replay buffer size
- Recovery from partial task completions

Resume Strategy:
    1. Checks for completed task checkpoints in continual/ directory
    2. Checks for intermediate checkpoints in checkpoints/ directory (partial task completions)
    3. Automatically resumes from the most recent checkpoint (completed or intermediate)
    4. Tracks training state to detect which task was being trained during interruptions

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
import time
import random
import pickle
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from constants import MAX_PROMPT_LEN, MAX_ANS_LEN, DEFAULT_TASK_CONFIGS, BASE_MODEL
from data_collator import DataCollator
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
    max_batch_size: Optional[int] = None  # Optional task-specific batch size

    def __str__(self) -> str:
        batch_info = (
            f", batch_size={self.max_batch_size}" if self.max_batch_size else ""
        )
        return f"{self.dataset_name} ({self.num_epochs} epochs{batch_info})"


class ExperienceReplayBuffer:
    """
    Manages replay buffer metadata for experience replay in continual learning.
    Generates replay datasets on-demand by loading and sampling from previous task datasets.
    Reduces catastrophic forgetting by rehearsing old examples during new task training.
    """

    def __init__(
        self, max_samples_per_task: int = 500, selection_strategy: str = "random"
    ):
        """
        Initialize the experience replay buffer.

        Args:
            max_samples_per_task: Maximum number of samples to select per task for replay
            selection_strategy: Strategy for selecting samples ('random', 'first')
        """
        self.max_samples_per_task = max_samples_per_task
        self.selection_strategy = selection_strategy
        # Only store metadata about completed tasks, not actual samples
        self.task_info: Dict[int, Dict] = {}  # task_id -> {"task_id": int, "dataset_name": str}

    def register_completed_task(self, task_id: int, dataset_name: str) -> None:
        """
        Register a completed task for future replay generation.
        Called at END of task (after checkpoint save).

        Args:
            task_id: ID of the completed task
            dataset_name: Name of the dataset (e.g., "MeetingBank", "FOMC")
        """
        self.task_info[task_id] = {
            "task_id": task_id,
            "dataset_name": dataset_name,
        }

        logger.info(
            f"✓ Registered task {task_id} ({dataset_name}) for replay buffer"
        )
        logger.info(
            f"  Replay buffer now tracks {len(self.task_info)} completed task(s)"
        )

    def get_replay_dataset(
        self,
        current_task_id: int,
        data_dir: Path,
        max_prompt_len: int,
        max_ans_len: int,
    ) -> Optional[Dataset]:
        """
        Generate replay dataset by loading and sampling from all previous tasks.
        Called at START of current task (before training).

        Args:
            current_task_id: ID of the current task being trained
            data_dir: Directory containing task datasets
            max_prompt_len: Maximum prompt length for formatting
            max_ans_len: Maximum answer length for formatting

        Returns:
            Combined dataset of replay samples from all previous tasks, or None if first task
        """
        if current_task_id == 0:
            logger.info("Task 0: No replay buffer (first task)")
            return None

        if len(self.task_info) == 0:
            logger.warning(
                f"No completed tasks registered in replay buffer for task {current_task_id}"
            )
            return None

        logger.info("=" * 80)
        logger.info(f"GENERATING REPLAY BUFFER FOR TASK {current_task_id}")
        logger.info("=" * 80)

        replay_datasets = []
        for prev_task_id in range(current_task_id):
            if prev_task_id not in self.task_info:
                logger.warning(
                    f"  Task {prev_task_id} not found in replay buffer metadata, skipping..."
                )
                continue

            # Get dataset name from metadata
            dataset_name = self.task_info[prev_task_id]["dataset_name"]

            logger.info(f"  Loading Task {prev_task_id} ({dataset_name}) for replay...")

            # Load the dataset from original source
            try:
                dataset = load_dataset(
                    "json",
                    data_files={
                        "train": str(data_dir / dataset_name / "train.json"),
                    },
                )["train"]

                logger.info(f"    Dataset loaded: {len(dataset)} samples")

                # Format (same as get_tokenized_dataset)
                def format_instruction(examples):
                    return {"prompt": examples["prompt"], "answer": examples["answer"]}

                MAP_BATCH_SIZE = 10000
                formatted = dataset.map(
                    format_instruction,
                    batched=True,
                    batch_size=MAP_BATCH_SIZE,
                )

                # Sample N examples according to selection strategy
                num_samples = min(self.max_samples_per_task, len(formatted))

                if self.selection_strategy == "random":
                    indices = random.sample(range(len(formatted)), num_samples)
                    selected = formatted.select(indices)
                elif self.selection_strategy == "first":
                    selected = formatted.select(range(num_samples))
                else:
                    # Default to random
                    indices = random.sample(range(len(formatted)), num_samples)
                    selected = formatted.select(indices)

                replay_datasets.append(selected)
                logger.info(f"    ✓ Sampled {num_samples} examples from Task {prev_task_id}")

            except Exception as e:
                logger.error(
                    f"    ✗ Failed to load dataset {dataset_name} for replay: {e}"
                )
                logger.error(f"    Skipping Task {prev_task_id} from replay buffer")
                continue

        if not replay_datasets:
            logger.warning(
                "  No replay datasets could be loaded. Training without replay."
            )
            return None

        # Combine all replay datasets
        combined = concatenate_datasets(replay_datasets)

        logger.info("=" * 80)
        logger.info(f"✓ REPLAY BUFFER READY")
        logger.info(f"  Total samples: {len(combined)} from {len(replay_datasets)} task(s)")
        logger.info(f"  Breakdown: {', '.join([f'Task {i}: {len(ds)}' for i, ds in enumerate(replay_datasets)])}")
        logger.info("=" * 80)

        return combined

    def save_metadata(self, path: Path) -> None:
        """
        Save only task metadata (dataset names), not actual samples.

        Args:
            path: Directory path to save the metadata
        """
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "max_samples_per_task": self.max_samples_per_task,
            "selection_strategy": self.selection_strategy,
            "task_info": self.task_info,
        }
        metadata_path = path / "replay_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"✓ Replay buffer metadata saved to {path}")
        logger.info(f"  Tracks {len(self.task_info)} completed task(s)")

    @classmethod
    def load_metadata(cls, path: Path) -> Optional["ExperienceReplayBuffer"]:
        """
        Load task metadata for replay generation.

        Args:
            path: Directory path containing the saved metadata

        Returns:
            ExperienceReplayBuffer with loaded metadata, or None if not found
        """
        if not path.exists():
            return None

        metadata_path = path / "replay_metadata.pkl"
        if not metadata_path.exists():
            logger.warning(f"No replay metadata found at {path}")
            return None

        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            buffer = cls(
                max_samples_per_task=metadata["max_samples_per_task"],
                selection_strategy=metadata["selection_strategy"],
            )
            buffer.task_info = metadata["task_info"]

            logger.info(
                f"✓ Replay buffer metadata loaded from {path} "
                f"({len(buffer.task_info)} tasks tracked)"
            )

            return buffer

        except Exception as e:
            logger.error(f"Failed to load replay buffer metadata: {e}")
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
        base_model_name: str = BASE_MODEL,
        output_base_dir: str = "./experiments",
        data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_5000",
        max_prompt_len: int = MAX_PROMPT_LEN,
        max_ans_len: int = MAX_ANS_LEN,
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
        self.current_base_model_path: str = base_model_name  # Tracks base model (original or merged)
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / "continual"  # Task checkpoints
        self.training_output_dir = (
            self.output_base_dir / "checkpoints"
        )  # Training artifacts
        self.replay_buffer_dir = (
            self.output_base_dir / "replay_buffer"
        )  # Replay buffer storage
        self.merged_models_dir = (
            self.output_base_dir / "merged_models"
        )  # Merged models (LoRA + base)
        self.data_dir = Path(data_dir)

        # Create cache dir using both prompt and answer lengths for uniqueness
        data_dir_name = self.data_dir.name
        cache_suffix = f"prompt{max_prompt_len}_ans{max_ans_len}"
        self.cache_dir = (
            Path(".cache") / "tokenized_datasets" / data_dir_name / cache_suffix
        )

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
                selection_strategy="random",
            )

        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        # Create output directories
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.merged_models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_training_state(self, task_id: int, task_name: str) -> None:
        """
        Save current training state to track partial task completions.

        Includes base_model_path to ensure intermediate checkpoint resume
        uses the correct base model.

        Args:
            task_id: Current task ID being trained
            task_name: Name of the current task
        """
        state_path = self.output_base_dir / "training_state.json"
        state = {
            "task_id": task_id,
            "task_name": task_name,
            "base_model_path": self.current_base_model_path,  # Track base model for resume
            "timestamp": str(time.time()),
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(
            f"Saved training state: task {task_id} ({task_name}), "
            f"base: {self.current_base_model_path}"
        )

    def load_training_state(self) -> Optional[Dict]:
        """
        Load training state to detect partial task completions.

        Returns:
            Dict with task_id and task_name, or None if no state file exists
        """
        state_path = self.output_base_dir / "training_state.json"
        if not state_path.exists():
            return None

        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load training state: {e}")
            return None

    def clear_training_state(self) -> None:
        """Clear training state file after successful checkpoint save."""
        state_path = self.output_base_dir / "training_state.json"
        if state_path.exists():
            state_path.unlink()
            logger.debug("Cleared training state")

    def cleanup_intermediate_checkpoints(self) -> None:
        """Remove intermediate checkpoints after task successfully completes."""
        if not self.training_output_dir.exists():
            return

        checkpoint_count = 0
        for item in self.training_output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    import shutil
                    shutil.rmtree(item)
                    checkpoint_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {item}: {e}")

        if checkpoint_count > 0:
            logger.info(f"Cleaned up {checkpoint_count} intermediate checkpoint(s)")

    def find_intermediate_checkpoint(self) -> Optional[Path]:
        """
        Find the latest intermediate checkpoint in the checkpoints directory.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        if not self.training_output_dir.exists():
            return None

        # Find all checkpoint directories (checkpoint-*, checkpoint-epoch-*)
        checkpoints = []
        for item in self.training_output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Verify it's a valid checkpoint
                if (item / "adapter_config.json").exists():
                    checkpoints.append(item)

        if not checkpoints:
            return None

        # Sort by modification time and return the latest
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found intermediate checkpoint: {latest}")
        return latest

    def find_last_checkpoint(
        self, tasks: List[Task]
    ) -> Tuple[Optional[int], Optional[Path], bool, Optional[str]]:
        """
        Find the last successful checkpoint, including intermediate checkpoints.

        Args:
            tasks: List of Task objects

        Returns:
            Tuple of (task_id, checkpoint_path, is_intermediate, base_model_path):
            - task_id: ID of the task to resume
            - checkpoint_path: Path to the checkpoint
            - is_intermediate: True if resuming from intermediate checkpoint (partial task)
            - base_model_path: Path to base model to use (original or merged)
        """
        logger.info("Searching for existing checkpoints...")

        # First, check for completed task checkpoints in continual/
        last_completed_task_id = None
        last_completed_checkpoint = None

        for task_id in range(len(tasks) - 1, -1, -1):
            task = tasks[task_id]
            checkpoint_path = self.output_dir / f"task_{task_id}_{task.dataset_name}"

            if (
                checkpoint_path.exists()
                and (checkpoint_path / "adapter_config.json").exists()
            ):
                logger.info(f"Found completed task checkpoint: {checkpoint_path}")
                last_completed_task_id = task_id
                last_completed_checkpoint = checkpoint_path
                break

        # Check for partial task completion via training state
        training_state = self.load_training_state()

        if training_state:
            partial_task_id = training_state.get("task_id")
            partial_task_name = training_state.get("task_name")

            # Only consider this if it's for a task after the last completed one
            if (
                last_completed_task_id is None
                or partial_task_id > last_completed_task_id
            ):
                # Check if there are intermediate checkpoints
                intermediate_checkpoint = self.find_intermediate_checkpoint()

                if intermediate_checkpoint:
                    # Validate checkpoint is complete
                    required_files = [
                        "adapter_config.json",
                        "trainer_state.json",
                        "optimizer.pt",
                    ]
                    is_valid = all(
                        (intermediate_checkpoint / f).exists() for f in required_files
                    )

                    if is_valid:
                        logger.info(
                            f"Found partial task completion: task {partial_task_id} ({partial_task_name})"
                        )
                        logger.info(
                            f"Intermediate checkpoint: {intermediate_checkpoint}"
                        )
                        # Get base model path from training state (for intermediate resume)
                        base_model_path = training_state.get(
                            "base_model_path", self.base_model_name
                        )
                        logger.info(
                            f"Will use base model from training state: {base_model_path}"
                        )
                        return partial_task_id, intermediate_checkpoint, True, base_model_path
                    else:
                        logger.warning(
                            f"Intermediate checkpoint {intermediate_checkpoint} is incomplete, ignoring"
                        )
                else:
                    # Stale training_state, no matching checkpoint
                    logger.warning(
                        f"Training state indicates task {partial_task_id} in progress, "
                        f"but no intermediate checkpoint found. Clearing stale state."
                    )
                    self.clear_training_state()
            elif partial_task_id == last_completed_task_id:
                # Task is already completed but training state wasn't cleared (stale state)
                logger.warning(
                    f"Training state indicates task {partial_task_id} in progress, "
                    f"but task checkpoint already exists in continual/. Clearing stale state."
                )
                self.clear_training_state()

        # Return the last completed task checkpoint
        if last_completed_task_id is not None:
            logger.info(
                f"Will resume from next task after completed task {last_completed_task_id}"
            )

            # Check for merged model from the completed task
            task = tasks[last_completed_task_id]
            merged_model_path = (
                self.merged_models_dir / f"task_{last_completed_task_id}_{task.dataset_name}"
            )

            if merged_model_path.exists():
                logger.info(f"Found merged model for next task: {merged_model_path}")
                base_model_path = str(merged_model_path)
            else:
                logger.warning(
                    f"No merged model found for task {last_completed_task_id}. "
                    f"Using original base model."
                )
                base_model_path = self.base_model_name

            return last_completed_task_id, last_completed_checkpoint, False, base_model_path

        logger.info("No existing checkpoints found")
        return None, None, False, None

    def setup_model_and_tokenizer(
        self, checkpoint_path: Optional[Path] = None, use_quantization: bool = True
    ) -> None:
        """
        Load model and tokenizer, optionally from a checkpoint.

        For progressive merging:
        - First task: Uses original base model
        - Later tasks: Uses merged model from previous task

        Args:
            checkpoint_path: Path to LoRA checkpoint to resume from
            use_quantization: Whether to use 4-bit quantization
        """
        logger.info("Setting up model and tokenizer...")
        logger.info(f"Base model path: {self.current_base_model_path}")

        # Configure quantization
        bnb_config = None
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer from current base model (original or merged)
        logger.info(f"Loading tokenizer from {self.current_base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.current_base_model_path, use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        # Load base model (original or merged from previous task)
        logger.info(f"Loading base model from {self.current_base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.current_base_model_path,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load from checkpoint or prepare new model
        if checkpoint_path:
            logger.info(f"Loading LoRA checkpoint from {checkpoint_path}")
            model = PeftModel.from_pretrained(
                base_model, str(checkpoint_path), is_trainable=True
            )
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
                r=16,
                lora_alpha=32,
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
                # use_rslora=True
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
        # if cache_path.exists():
        if False:
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
            # Keep both prompt and answer fields for DataCollator
            # DataCollator will handle tokenization and proper label masking
            return {"prompt": examples["prompt"], "answer": examples["answer"]}

        logger.info(f"Formatting dataset: {dataset_name}")
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        logger.info(f"Train size: {len(dataset['train'])}")
        MAP_BATCH_SIZE = 10000  # Batch size for dataset.map() operations
        formatted_dataset = dataset.map(
            format_instruction,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            # Don't remove any columns - DataCollator needs both prompt and answer
        )
        logger.info(f"Formatting completed successfully")

        # DataCollator will handle tokenization on-the-fly during training
        # This is slower but ensures correct label alignment (no context-dependent tokenization bugs)

        # Save to cache for future use
        logger.info(f"Saving formatted dataset to cache: {cache_path}")
        formatted_dataset.save_to_disk(str(cache_path))

        logger.info(f"Dataset {dataset_name} prepared successfully")
        return formatted_dataset

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
        replay_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[Path] = None,
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
            resume_from_checkpoint: Optional path to intermediate checkpoint to resume training from
        """
        assert (
            self.model is not None
        ), "Model not initialized. Call setup_model_and_tokenizer first."
        assert (
            self.tokenizer is not None
        ), "Tokenizer not initialized. Call setup_model_and_tokenizer first."

        # Use task-specific batch size if provided, otherwise use default
        # effective_batch_size = batch_size if batch_size is not None else self.batch_size
        effective_batch_size = min(batch_size or self.batch_size, self.batch_size)

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
            logger.info(
                f"  - Current task: {current_task_size} samples (100% of task data)"
            )
            logger.info(f"  - Replay buffer: {replay_size} samples (added on top)")

            # Concatenate full datasets and shuffle for better mixing
            train_dataset = concatenate_datasets(
                [train_dataset, replay_dataset]
            ).shuffle(seed=42)

            logger.info(f"Total training dataset size: {len(train_dataset)} samples")
            logger.info(f"  ({current_task_size} new + {replay_size} replay)")
            logger.info("=" * 60)
        else:
            logger.info(f"Training on current task only ({len(train_dataset)} samples)")

        logger.info(
            f"Starting training for {num_epochs} epochs with batch size {effective_batch_size}..."
        )

        training_args = TrainingArguments(
            output_dir=str(self.training_output_dir),
            per_device_train_batch_size=effective_batch_size,
            gradient_accumulation_steps=4,
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
            remove_unused_columns=False,  # Keep 'prompt' and 'answer' fields for DataCollator
        )

        # Use original DataCollator for correct label alignment
        # This fixes the context-dependent tokenization bug in the custom collator
        data_collator = DataCollator(
            tokenizer=self.tokenizer,
            max_prompt_len=self.max_prompt_len,
            max_ans_len=self.max_ans_len,
            pad_to_multiple_of=1,
            inference=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Resume from checkpoint if provided (restores optimizer, scheduler, epoch counter)
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
        else:
            trainer.train()
        logger.info("Training completed")

    def save_checkpoint(self, task_id: int, dataset_name: str) -> Path:
        """
        Save model checkpoint including LoRA adapter and merged model.

        This method:
        1. Saves LoRA adapter to continual/task_{id}_{dataset}/
        2. Merges LoRA into base model and saves to merged_models/task_{id}_{dataset}/
        3. Updates current_base_model_path for next task

        Args:
            task_id: Task ID number
            dataset_name: Name of the dataset

        Returns:
            Path to saved LoRA checkpoint
        """
        assert self.model is not None, "Model not initialized."
        assert self.tokenizer is not None, "Tokenizer not initialized."

        # Step 1: Save LoRA adapter (existing behavior)
        checkpoint_path = self.output_dir / f"task_{task_id}_{dataset_name}"
        logger.info(f"Saving LoRA checkpoint to {checkpoint_path}")

        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        logger.info("✓ LoRA checkpoint saved")

        # Step 2: Merge and save merged model (NEW)
        logger.info(f"Merging LoRA into base model for task {task_id}...")
        merged_path = self.merge_and_save_model(task_id, dataset_name)

        # Step 3: Update base model path for next task (NEW)
        if merged_path:
            logger.info(f"✓ Merged model saved to {merged_path}")
            # Update base model path for next task
            self.current_base_model_path = str(merged_path)
            logger.info(
                f"Next task will use merged model: {self.current_base_model_path}"
            )
        else:
            logger.warning(
                f"⚠ Merge failed for task {task_id}. Next task will use "
                f"current base: {self.current_base_model_path}"
            )

        # Step 4: Clear training state (existing behavior)
        self.clear_training_state()

        logger.info(f"Checkpoint saved successfully")
        return checkpoint_path

    def merge_and_save_model(
        self, task_id: int, dataset_name: str
    ) -> Optional[Path]:
        """
        Merge current LoRA adapter into base model and save merged model.

        This implements progressive merging: each task's merged model becomes
        the base for the next task.

        Process:
        1. Load base model in full precision (bfloat16, no quantization)
        2. Load current LoRA adapter from the saved checkpoint
        3. Merge LoRA weights into base model
        4. Save merged model to merged_models/task_{id}_{dataset}/
        5. Save tokenizer
        6. Clean up GPU memory

        Args:
            task_id: Current task ID
            dataset_name: Current task dataset name

        Returns:
            Path to saved merged model, or None if merge failed
        """
        try:
            merged_model_path = self.merged_models_dir / f"task_{task_id}_{dataset_name}"
            logger.info("=" * 80)
            logger.info(f"MERGING LORA ADAPTER FOR TASK {task_id}")
            logger.info(f"Base model: {self.current_base_model_path}")
            logger.info(f"Output path: {merged_model_path}")
            logger.info("=" * 80)

            # Step 1: Load base model in full precision (no quantization)
            logger.info("Loading base model in full precision for merging...")
            base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                self.current_base_model_path,
                torch_dtype=torch.bfloat16,  # Always bfloat16 for merged models
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✓ Base model loaded")

            # Step 2: Load current LoRA adapter
            # Get LoRA checkpoint path for this task
            lora_checkpoint_path = self.output_dir / f"task_{task_id}_{dataset_name}"
            logger.info(f"Loading LoRA adapter from {lora_checkpoint_path}")

            # Load LoRA onto base model
            model_with_lora = PeftModel.from_pretrained(
                base_model_for_merge,
                str(lora_checkpoint_path),
                is_trainable=False,  # Not training, just merging
            )
            logger.info("✓ LoRA adapter loaded")

            # Step 3: Merge LoRA weights into base model
            logger.info("Merging LoRA weights into base model...")
            merged_model = model_with_lora.merge_and_unload()
            logger.info("✓ Merge complete")

            # Step 4: Save merged model
            logger.info(f"Saving merged model to {merged_model_path}")
            merged_model_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_model_path))
            logger.info("✓ Merged model saved")

            # Step 5: Save tokenizer (copy from current tokenizer)
            logger.info("Saving tokenizer...")
            self.tokenizer.save_pretrained(str(merged_model_path))
            logger.info("✓ Tokenizer saved")

            # Step 6: Clean up GPU memory
            del base_model_for_merge
            del model_with_lora
            del merged_model
            torch.cuda.empty_cache()

            logger.info("=" * 80)
            logger.info(f"✓ MERGE SUCCESSFUL FOR TASK {task_id}")
            logger.info(f"Merged model size: ~6.4GB")
            logger.info(f"This will be the base model for task {task_id + 1}")
            logger.info("=" * 80)

            return merged_model_path

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"✗ MERGE FAILED FOR TASK {task_id}")
            logger.error(f"Error: {type(e).__name__}: {e}")
            logger.error("=" * 80)
            logger.error(
                "LoRA adapter is still saved. Next task will use current base."
            )
            return None

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
        is_partial_task_resume = False
        base_model_for_next_task = self.base_model_name  # Default: original base

        if auto_resume:
            (
                last_task_id,
                checkpoint_path,
                is_intermediate,
                base_model_path,
            ) = self.find_last_checkpoint(tasks)

            if last_task_id is not None:
                is_partial_task_resume = is_intermediate

                # Update current base model path from checkpoint info
                if base_model_path:
                    base_model_for_next_task = base_model_path
                    logger.info(f"Will use base model: {base_model_for_next_task}")

                if is_intermediate:
                    # Resuming from an intermediate checkpoint (partial task)
                    start_task_id = last_task_id
                    logger.info("=" * 80)
                    logger.info(f"RESUMING FROM INTERMEDIATE CHECKPOINT")
                    logger.info(f"Partial task: {last_task_id} ({tasks[last_task_id]})")
                    logger.info(f"Will continue training task: {start_task_id}")
                    logger.info(f"Checkpoint: {checkpoint_path}")
                    logger.info("=" * 80)
                else:
                    # Resuming from a completed task checkpoint
                    start_task_id = last_task_id + 1
                    logger.info("=" * 80)
                    logger.info(f"RESUMING FROM CHECKPOINT")
                    logger.info(
                        f"Last completed task: {last_task_id} ({tasks[last_task_id]})"
                    )
                    logger.info(f"Resuming from task: {start_task_id}")
                    logger.info("=" * 80)

                # Load replay buffer metadata if resuming and replay is enabled
                if self.use_experience_replay and self.replay_buffer is not None:
                    loaded_buffer = ExperienceReplayBuffer.load_metadata(
                        self.replay_buffer_dir
                    )
                    if loaded_buffer is not None:
                        self.replay_buffer = loaded_buffer
                        logger.info(
                            f"Replay buffer metadata loaded: {len(self.replay_buffer.task_info)} tasks tracked"
                        )
                    else:
                        logger.warning(
                            "Could not load replay buffer metadata, starting fresh buffer"
                        )

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

        # Set current base model path for first task
        self.current_base_model_path = base_model_for_next_task

        # Setup model for first task
        self.setup_model_and_tokenizer(
            checkpoint_path=checkpoint_path, use_quantization=use_quantization
        )

        # Train on remaining tasks
        for task_id in range(start_task_id, len(tasks)):
            task = tasks[task_id]

            # IMPORTANT: For task_id > start_task_id, reload model with new base
            # This implements progressive merging: each task uses the merged model from previous task
            if task_id > start_task_id:
                logger.info("=" * 80)
                logger.info(f"RELOADING MODEL FOR TASK {task_id}")
                logger.info(f"Base model: {self.current_base_model_path}")
                logger.info("=" * 80)

                # Clean up previous model to free memory
                if self.model is not None:
                    del self.model
                    torch.cuda.empty_cache()
                    logger.info("Previous model deleted, GPU memory cleared")

                # Load new base (merged model from previous task) + fresh LoRA
                self.setup_model_and_tokenizer(
                    checkpoint_path=None,  # Fresh LoRA adapter
                    use_quantization=use_quantization,
                )
                logger.info(f"Model reloaded with base: {self.current_base_model_path}")

            logger.info("=" * 80)
            logger.info(f"TASK {task_id}/{len(tasks)-1}: {task}")
            logger.info("=" * 80)

            try:
                # Save training state before starting task (for recovery from intermediate checkpoints)
                self.save_training_state(task_id, task.dataset_name)

                # Prepare dataset
                tokenized_dataset = self.get_tokenized_dataset(task.dataset_name)

                # Generate replay dataset from previous tasks if replay is enabled
                replay_dataset = None
                if self.use_experience_replay and self.replay_buffer is not None:
                    replay_dataset = self.replay_buffer.get_replay_dataset(
                        current_task_id=task_id,
                        data_dir=self.data_dir,
                        max_prompt_len=self.max_prompt_len,
                        max_ans_len=self.max_ans_len,
                    )

                # Determine if we should resume from an intermediate checkpoint
                # Only resume for the first task if we're continuing a partial task
                resume_checkpoint = None
                if (
                    task_id == start_task_id
                    and is_partial_task_resume
                    and checkpoint_path
                ):
                    # Find the latest intermediate checkpoint in checkpoints/ directory
                    resume_checkpoint = self.find_intermediate_checkpoint()
                    if resume_checkpoint:
                        logger.info(
                            f"Will resume training from intermediate checkpoint: {resume_checkpoint}"
                        )

                # Train (use task-specific batch size if provided)
                self.train_model(
                    tokenized_dataset,
                    num_epochs=task.num_epochs,
                    batch_size=task.max_batch_size,
                    replay_dataset=replay_dataset,
                    resume_from_checkpoint=resume_checkpoint,
                )

                # Register completed task for future replay generation
                # This ensures replay buffer is always consistent with completed checkpoints
                if self.use_experience_replay and self.replay_buffer is not None:
                    self.replay_buffer.register_completed_task(
                        task_id=task_id,
                        dataset_name=task.dataset_name,
                    )
                    # Save replay buffer metadata after each task
                    self.replay_buffer.save_metadata(self.replay_buffer_dir)

                # Save checkpoint (marks task as officially complete)
                self.save_checkpoint(task_id, task.dataset_name)
                self.cleanup_intermediate_checkpoints()

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
    parser.add_argument(
        "--data-dir",
        type=str,
        default="TRACE-Benchmark/LLM-CL-Benchmark_5000",
        help="Dataset directory (default: LLM-CL-Benchmark_5000). Use LLM-CL-Benchmark_500 for quick testing",
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
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        choices=DEFAULT_TASK_CONFIGS.keys(),
        help="Can be passed multiple times.",
    )

    args = parser.parse_args()

    # Define all tasks
    # Note: MeetingBank uses batch_size=8 due to long sequences (200+ tokens)
    # to avoid OOM errors. Other datasets can use the default batch size.
    # H100
    # tasks = [
    #     Task(dataset_name='C-STANCE', num_epochs=5),
    #     Task(dataset_name="FOMC", num_epochs=3),
    #     Task(dataset_name='MeetingBank', num_epochs=7, batch_size=32),
    #     Task(dataset_name='Py150', num_epochs=5, batch_size=32),
    #     Task(dataset_name='ScienceQA', num_epochs=3, batch_size=64),
    #     Task(dataset_name='NumGLUE-cm', num_epochs=5),
    #     Task(dataset_name='NumGLUE-ds', num_epochs=5),
    #     Task(dataset_name='20Minuten', num_epochs=7, batch_size=32),
    # ]
    # H200
    # tasks = [
    #     # Task(dataset_name='C-STANCE', num_epochs=5),
    #     Task(dataset_name='MeetingBank', num_epochs=7, max_batch_size=64), # 7 epochs
    #     Task(dataset_name="FOMC", num_epochs=3),
    #     Task(dataset_name='Py150', num_epochs=5, max_batch_size=64),
    #     Task(dataset_name='ScienceQA', num_epochs=3, max_batch_size=128),
    #     Task(dataset_name='NumGLUE-cm', num_epochs=5),
    #     Task(dataset_name='NumGLUE-ds', num_epochs=5),
    #     Task(dataset_name='20Minuten', num_epochs=7, max_batch_size=64),
    # ]
    tasks = []
    for name in args.task:
        cfg = DEFAULT_TASK_CONFIGS[name]
        tasks.append(Task(dataset_name=name, **cfg))

    logger.info("=" * 80)
    logger.info("CONTINUAL LEARNING TRAINING")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Tasks: {[task.dataset_name for task in tasks]}")
    logger.info(f"Dataset directory: {args.data_dir}")
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
    logger.info("--- Experience Replay Configuration ---")
    if not args.no_replay:
        logger.info(f"Experience Replay: ENABLED (Additive Mode)")
        logger.info(f"  - Samples stored per task: {args.replay_samples_per_task}")
        logger.info(f"  - Mode: All replay samples added on top of full task data")
        logger.info(
            f"  - Growth: Task dataset increases by ~{args.replay_samples_per_task} samples per task"
        )
    else:
        logger.info(f"Experience Replay: DISABLED")
        logger.info(
            f"  ⚠️  Warning: Catastrophic forgetting will be higher without replay!"
        )
    logger.info("=" * 80)

    # Initialize trainer
    trainer = ContinualLearningTrainer(
        output_base_dir=args.output_base_dir,
        data_dir=args.data_dir,
        base_model_name=BASE_MODEL,
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
