"""
Full Model Fine-Tuning for Continual Learning

This script performs full fine-tuning where ALL model parameters are trained.

Usage:
    python training_full.py --task MeetingBank --task FOMC
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from training import (
    ContinualLearningTrainer,
    Task,
    DEFAULT_TASK_CONFIGS,
)
from constants import MAX_PROMPT_LEN, MAX_ANS_LEN, BASE_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training_full.log")],
)
logger = logging.getLogger(__name__)


class FullFineTuningTrainer(ContinualLearningTrainer):
    """
    Full fine-tuning trainer for continual learning.

    Inherits most functionality from ContinualLearningTrainer but overrides
    model initialization and checkpoint detection for full fine-tuning instead of LoRA.
    """

    def _is_valid_full_checkpoint(self, path: Path) -> bool:
        """
        Check if path contains a valid full fine-tuning checkpoint.

        Args:
            path: Path to potential checkpoint directory

        Returns:
            True if valid full model checkpoint, False otherwise
        """
        # Check for config file
        if not (path / "config.json").exists():
            return False

        # Check for model weights (any of these formats)
        has_weights = (
            (path / "pytorch_model.bin").exists() or
            (path / "model.safetensors").exists() or
            len(list(path.glob("pytorch_model-*.bin"))) > 0  # Sharded models
        )

        return has_weights

    def find_intermediate_checkpoint(self) -> Optional[Path]:
        """
        Find the latest intermediate checkpoint in the checkpoints directory.

        Overrides parent method to check for full model files instead of adapter_config.json.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        if not self.training_output_dir.exists():
            return None

        # Find all checkpoint directories (checkpoint-*, checkpoint-epoch-*)
        checkpoints = []
        for item in self.training_output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Verify it's a valid full model checkpoint
                if self._is_valid_full_checkpoint(item):
                    checkpoints.append(item)

        if not checkpoints:
            return None

        # Sort by modification time and return the latest
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found intermediate checkpoint: {latest}")
        return latest

    def find_last_checkpoint(
        self, tasks: List[Task]
    ) -> Tuple[Optional[int], Optional[Path], bool, str]:
        """
        Find the last successful checkpoint, including intermediate checkpoints.

        Overrides parent method to check for full model files instead of adapter_config.json.

        Args:
            tasks: List of Task objects

        Returns:
            Tuple of (task_id, checkpoint_path, is_intermediate):
            - task_id: ID of the task to resume
            - checkpoint_path: Path to the checkpoint
            - is_intermediate: True if resuming from intermediate checkpoint (partial task)
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
                and self._is_valid_full_checkpoint(checkpoint_path)
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
            if (last_completed_task_id is None or partial_task_id > last_completed_task_id):
                # Check if there are intermediate checkpoints
                intermediate_checkpoint = self.find_intermediate_checkpoint()

                if intermediate_checkpoint:
                    # Validate checkpoint is complete
                    required_files = ["config.json", "trainer_state.json", "optimizer.pt"]
                    is_valid = all((intermediate_checkpoint / f).exists() for f in required_files)

                    if is_valid:
                        logger.info(f"Found partial task completion: task {partial_task_id} ({partial_task_name})")
                        logger.info(f"Intermediate checkpoint: {intermediate_checkpoint}")
                        return partial_task_id, intermediate_checkpoint, True, BASE_MODEL
                    else:
                        logger.warning(f"Intermediate checkpoint {intermediate_checkpoint} is incomplete, ignoring")
                else:
                    # Stale training_state, no matching checkpoint
                    logger.warning(f"Training state indicates task {partial_task_id} in progress, "
                                  f"but no intermediate checkpoint found. Clearing stale state.")
                    self.clear_training_state()
            elif partial_task_id == last_completed_task_id:
                # Task is already completed but training state wasn't cleared (stale state)
                logger.warning(f"Training state indicates task {partial_task_id} in progress, "
                              f"but task checkpoint already exists in continual/. Clearing stale state.")
                self.clear_training_state()

        # Return the last completed task checkpoint
        if last_completed_task_id is not None:
            logger.info(f"Will resume from next task after completed task {last_completed_task_id}")
            return last_completed_task_id, last_completed_checkpoint, False, BASE_MODEL

        logger.info("No existing checkpoints found")
        return None, None, False, BASE_MODEL

    def setup_model_and_tokenizer(
        self, checkpoint_path: Optional[Path] = None, use_quantization: bool = True
    ) -> None:
        """
        Load model and tokenizer for full fine-tuning.

        Overrides parent method to load full model instead of LoRA adapters.

        Args:
            checkpoint_path: Path to full model checkpoint to resume from
            use_quantization: Whether to use 4-bit quantization
        """
        logger.info("=" * 80)
        logger.info("Setting up model and tokenizer for FULL FINE-TUNING")
        logger.info("All model parameters will be trained (no LoRA adapters)")
        logger.info("=" * 80)

        # Configure quantization (optional)
        bnb_config = None
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Quantization: ENABLED (4-bit)")
        else:
            logger.info("Quantization: DISABLED (full precision)")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        # Load model (from checkpoint or base model)
        if checkpoint_path:
            logger.info(f"Loading full fine-tuned model from {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            logger.info(f"Loading base model from {self.base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Prepare model for training
        if use_quantization:
            logger.info("Preparing quantized model for training...")
            model = prepare_model_for_kbit_training(model)
        else:
            # Enable all parameters for training (full fine-tuning)
            logger.info("Enabling all parameters for training...")
            for param in model.parameters():
                param.requires_grad = True
            model.gradient_checkpointing_enable()

        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = 100 * trainable_params / total_params

        logger.info("=" * 80)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {trainable_percentage:.2f}%")
        logger.info("=" * 80)

        self.model = model
        logger.info("Model and tokenizer setup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full Model Fine-Tuning for Continual Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full fine-tuning with auto-resume and experience replay (default)
  python training_full.py --task MeetingBank --task FOMC
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
        default="./experiments_full",
        help="Base directory for outputs (creates 'continual' and 'checkpoints' subdirs, default: ./experiments_full)",
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
        help="Enable 4-bit quantization (useful for GPUs with limited VRAM)",
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
        help="Task to train on. Can be passed multiple times.",
    )

    args = parser.parse_args()

    tasks = []
    for name in args.task:
        cfg = DEFAULT_TASK_CONFIGS[name]
        tasks.append(Task(dataset_name=name, num_epochs=cfg['num_epochs'], max_batch_size=cfg.get('batch_size')))

    logger.info("=" * 80)
    logger.info("FULL FINE-TUNING - CONTINUAL LEARNING")
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
        logger.info(f"  - Growth: Task dataset increases by ~{args.replay_samples_per_task} samples per task")
    else:
        logger.info(f"Experience Replay: DISABLED")
        logger.info(f"    ️  Warning: Catastrophic forgetting will be higher without replay!")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = FullFineTuningTrainer(
        output_base_dir=args.output_base_dir,
        data_dir=args.data_dir,
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
