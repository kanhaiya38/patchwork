"""
Model Loading Utilities

Centralized model and tokenizer loading following DRY principle.
Handles both base models and LoRA checkpoints with consistent configuration.
"""

import logging
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading of base models and LoRA checkpoints.

    Single Responsibility: Model/tokenizer loading and configuration
    """

    def __init__(
        self,
        base_model_name: str = "openlm-research/open_llama_3b_v2",
        use_quantization: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize model loader with configuration.

        Args:
            base_model_name: HuggingFace model identifier
            use_quantization: Whether to use 4-bit quantization
            device: Device to load model on
        """
        self.base_model_name = base_model_name
        self.use_quantization = use_quantization
        self.device = device
        self._quantization_config = self._create_quantization_config()

    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create quantization configuration if enabled.

        Returns:
            BitsAndBytesConfig or None
        """
        if not self.use_quantization:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load and configure tokenizer.

        Returns:
            Configured tokenizer
        """
        logger.info(f"Loading tokenizer from {self.base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    def load_base_model(self) -> AutoModelForCausalLM:
        """
        Load base model without LoRA adapters.

        Returns:
            Base model
        """
        logger.info(f"Loading base model: {self.base_model_name}")
        logger.info(f"Quantization: {'enabled (4-bit)' if self.use_quantization else 'disabled'}")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self._quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Base model loaded successfully")
        return model

    def load_checkpoint(self, checkpoint_path: str) -> PeftModel:
        """
        Load model with LoRA checkpoint.

        Args:
            checkpoint_path: Path to LoRA checkpoint directory

        Returns:
            Model with LoRA adapters loaded
        """
        logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")

        # Load base model first
        base_model = self.load_base_model()

        # Load LoRA checkpoint
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

        logger.info("Checkpoint loaded successfully")
        return model

    def load_model_and_tokenizer(
        self,
        checkpoint_path: Optional[str] = None,
        eval_mode: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model (with optional checkpoint) and tokenizer.

        Args:
            checkpoint_path: Optional path to LoRA checkpoint
            eval_mode: Whether to set model to eval mode

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = self.load_tokenizer()

        if checkpoint_path:
            model = self.load_checkpoint(checkpoint_path)
        else:
            model = self.load_base_model()

        if eval_mode:
            model.eval()

        return model, tokenizer
