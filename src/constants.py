"""
Constants for continual learning training and validation.

These values should be consistent across training and validation to ensure
proper model evaluation.
"""

# Maximum length for input prompts (in tokens)
MAX_PROMPT_LEN = 1024

# Maximum length for generated answers (in tokens)
MAX_ANS_LEN = 512

# Total maximum sequence length
MAX_SEQUENCE_LEN = MAX_PROMPT_LEN + MAX_ANS_LEN


BASE_MODEL = "openlm-research/open_llama_3b_v2"


DEFAULT_TASK_CONFIGS = {
    "C-STANCE": dict(num_epochs=5),
    "MeetingBank": dict(num_epochs=7, max_batch_size=64),
    "FOMC": dict(
        num_epochs=3,
    ),
    "Py150": dict(num_epochs=5, max_batch_size=64),
    "ScienceQA": dict(
        num_epochs=3,
    ),
    "NumGLUE-cm": dict(
        num_epochs=5,
    ),
    "NumGLUE-ds": dict(
        num_epochs=5,
    ),
    "20Minuten": dict(
        num_epochs=7,
    ),
}
