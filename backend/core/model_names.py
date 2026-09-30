"""
Custom model names.

A custom model is stored on the models volume under {user_id}/{model_name}/, so its name is
also a folder name. Names are limited to letters, digits, '_' and '-', so no name can form
a path that reaches another folder (such as '../12/their-model').
"""

import re

MODEL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def is_valid_model_name(name: str) -> bool:
    """Whether a name is allowed for a custom model."""
    return isinstance(name, str) and re.fullmatch(MODEL_NAME_PATTERN, name) is not None


def model_folder(user_id: int, model_name: str) -> str:
    """
    A user's model folder on the models volume, relative to the volume root.

    Raises ValueError for a name that isn't a valid model name, so a bad name can never
    read, write, or delete outside the owner's folder.
    """
    if not isinstance(user_id, int) or not is_valid_model_name(model_name):
        raise ValueError(f"Invalid model name: {model_name!r}")
    return f"{user_id}/{model_name}"
