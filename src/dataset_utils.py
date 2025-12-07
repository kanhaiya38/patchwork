"""
Dataset Utilities
"""

import logging
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DatasetLoader:
    def __init__(
        self,
        data_dir: str = "TRACE-Benchmark/LLM-CL-Benchmark_500",
        max_length: int = 512
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length

    def load_dataset_splits(
        self,
        dataset_name: str,
        splits: list = None
    ) -> Dict[str, Dataset]:
        if splits is None:
            splits = ["train", "test", "eval"]

        logger.info(f"Loading dataset: {dataset_name}")

        dataset_path = self.data_dir / dataset_name
        data_files = {
            split: str(dataset_path / f"{split}.json")
            for split in splits
        }

        dataset = load_dataset("json", data_files=data_files)
        logger.info(f"Dataset {dataset_name} loaded successfully")

        return dataset

    def load_eval_dataset(self, dataset_name: str) -> Dataset:
        logger.info(f"Loading eval dataset: {dataset_name}")

        dataset_path = self.data_dir / dataset_name
        dataset = load_dataset(
            "json",
            data_files={"eval": str(dataset_path / "eval.json")}
        )

        return dataset["eval"]

    @staticmethod
    def format_instruction_dataset(dataset: Dataset) -> Dataset:
        logger.info("Formatting dataset")

        def format_instruction(examples):
            texts = [
                prompt + answer
                for prompt, answer in zip(examples["prompt"], examples["answer"])
            ]
            return {"text": texts}

        return dataset.map(
            format_instruction,
            batched=True,
            remove_columns=["prompt", "answer"]
        )

    @staticmethod
    def tokenize_dataset(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> Dataset:
        logger.info("Tokenizing dataset")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length
            )

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

    def prepare_training_dataset(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer
    ) -> Dataset:
        # Load all splits
        dataset = self.load_dataset_splits(dataset_name)

        # Format and tokenize
        formatted_dataset = self.format_instruction_dataset(dataset)
        tokenized_dataset = self.tokenize_dataset(
            formatted_dataset,
            tokenizer,
            self.max_length
        )

        logger.info(f"Dataset {dataset_name} prepared successfully")
        return tokenized_dataset
