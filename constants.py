"""
Constants for continual learning training and validation.

These values should be consistent across training and validation to ensure
proper model evaluation. Do not modify these without understanding the implications.
"""

# Maximum length for input prompts (in tokens)
# This determines how much context the model can see during training/validation
MAX_PROMPT_LEN = 1024

# Maximum length for generated answers (in tokens)
# This determines the maximum length of model outputs
MAX_ANS_LEN = 512

# Total maximum sequence length
MAX_SEQUENCE_LEN = MAX_PROMPT_LEN + MAX_ANS_LEN
