"""
Validation/Evaluation Module for Continual Learning

This module handles model validation during and after training.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for evaluation modules
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.evaluations import (
    eval_ScienceQA,
    eval_MeetingBank,
    eval_PapyrusF,
    eval_CStance,
    eval_Py150,
    eval_FOMC,
    eval_NumGLUE_cm,
    eval_NumGLUE_ds,
    eval_20Minuten
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """Handles model validation and evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        max_ans_len: int = 256,
        temperature: float = 0.1,
        inference_batch_size: int = 4,
        device: str = "cuda"
    ):
        """
        Initialize the validator.

        Args:
            model: The model to validate
            tokenizer: Tokenizer for the model
            max_ans_len: Maximum answer length for generation
            temperature: Temperature for generation
            inference_batch_size: Batch size for inference
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_ans_len = max_ans_len
        self.temperature = temperature
        self.inference_batch_size = inference_batch_size
        self.device = device

    def predict(self, dataloader) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate predictions for a dataset.

        Args:
            dataloader: DataLoader for the evaluation dataset

        Returns:
            Tuple of (source_sequences, predicted_sequences, ground_truths)
        """
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        self.model.eval()

        progress_bar = tqdm(total=len(dataloader), desc="Validating", leave=True)

        for step, batch in enumerate(dataloader):
            # Extract sources and ground truths
            sources_sequences += batch['sources']
            ground_truths += batch['gts']

            # Remove non-tensor items from batch
            del batch['sources']
            del batch['gts']

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            prompt_len = batch['input_ids'].shape[1]

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_description(f"Step {step}", refresh=False)

            with torch.no_grad():
                # Generate predictions with greedy decoding (deterministic)
                generate_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=self.max_ans_len,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,  # Greedy decoding - picks highest probability token
                    num_return_sequences=1,
                    use_cache=True
                )

            # Decode only the generated part (not the prompt)
            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            predicted_sequences += sequences

        progress_bar.close()

        return sources_sequences, predicted_sequences, ground_truths

    def evaluate_predictions(
        self,
        dataset_name: str,
        sources: List[str],
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions using the appropriate evaluation function.

        Args:
            dataset_name: Name of the dataset
            sources: Source sequences
            predictions: Predicted sequences
            ground_truths: Ground truth sequences

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating predictions for {dataset_name}")

        # Map dataset names to evaluation functions
        if dataset_name == "ScienceQA":
            evaluation_result = eval_ScienceQA.eval(predictions, ground_truths)
        elif dataset_name == "MeetingBank":
            evaluation_result = eval_MeetingBank.eval(predictions, ground_truths)
        elif dataset_name == "C-STANCE":
            evaluation_result = eval_CStance.eval(predictions, ground_truths)
        elif dataset_name == "Papyrus-f":
            evaluation_result = eval_PapyrusF.eval(predictions, ground_truths)
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
            logger.warning(f"No evaluation function found for {dataset_name}")
            evaluation_result = {}

        return evaluation_result

    def save_validation_results(
        self,
        evaluation_result: Dict[str, Any],
        sources: List[str],
        predictions: List[str],
        ground_truths: List[str],
        output_path: str,
        task_id: int,
        dataset_name: str
    ) -> None:
        """
        Save validation results to a JSON file.

        Args:
            evaluation_result: Dictionary with evaluation metrics
            sources: Source sequences
            predictions: Predicted sequences
            ground_truths: Ground truth sequences
            output_path: Directory to save results
            task_id: Task ID number
            dataset_name: Name of the dataset
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Prepare data
        results_data = {
            "eval": evaluation_result,
            "prompts": sources,
            "results": predictions,
            "labels": ground_truths
        }

        # Save to JSON file
        output_file = os.path.join(
            output_path,
            f"validation-task_{task_id}-{dataset_name}.json"
        )

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Validation results saved to {output_file}")
        logger.info(f"Metrics: {evaluation_result}")


def create_dataloader_from_dataset(
    dataset: Dataset,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
) -> DataLoader:
    """
    Create a DataLoader from a HuggingFace Dataset for evaluation.

    Args:
        dataset: HuggingFace Dataset with 'prompt' and 'answer' fields
        tokenizer: Tokenizer to use
        batch_size: Batch size for the DataLoader
        max_length: Maximum sequence length

    Returns:
        DataLoader ready for evaluation
    """
    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "sources": prompts,
            "gts": answers,
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


def run_validation(
    model,
    tokenizer,
    eval_dataloader,
    dataset_name: str,
    task_id: int,
    output_path: str = "./validation-results",
    max_ans_len: int = 256,
    temperature: float = 0.1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Convenience function to run validation.

    Args:
        model: Model to validate
        tokenizer: Tokenizer
        eval_dataloader: DataLoader for evaluation data
        dataset_name: Name of the dataset
        task_id: Task ID number
        output_path: Directory to save results
        max_ans_len: Maximum answer length
        temperature: Generation temperature
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    validator = ModelValidator(
        model=model,
        tokenizer=tokenizer,
        max_ans_len=max_ans_len,
        temperature=temperature,
        device=device
    )

    # Generate predictions
    sources, predictions, ground_truths = validator.predict(eval_dataloader)

    # Evaluate
    evaluation_result = validator.evaluate_predictions(
        dataset_name=dataset_name,
        sources=sources,
        predictions=predictions,
        ground_truths=ground_truths
    )

    # Save results
    validator.save_validation_results(
        evaluation_result=evaluation_result,
        sources=sources,
        predictions=predictions,
        ground_truths=ground_truths,
        output_path=output_path,
        task_id=task_id,
        dataset_name=dataset_name
    )

    return evaluation_result
