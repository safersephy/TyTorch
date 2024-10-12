import inspect
import pickle
from typing import Any, Dict

import torch
from loguru import logger


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("using cuda")
    else:
        device = "cpu"
        logger.info("using cpu")
    return device


def save_params_to_disk(params: Dict[str, Any], file_path: str) -> None:
    """Save the params dictionary (with class references) to a pickle file on disk."""
    try:
        with open(file_path, "wb") as pickle_file:
            pickle.dump(params, pickle_file)
        print(f"Parameters saved to {file_path}")
    except Exception as e:
        print(f"Error saving params to disk: {e}")


def load_params_from_disk(file_path: str) -> Dict[str, Any]:
    """Load the params dictionary (with class references) from a pickle file on disk."""
    try:
        with open(file_path, "rb") as pickle_file:
            params = pickle.load(pickle_file)
        print(f"Parameters loaded from {file_path}")
        return params
    except Exception as e:
        print(f"Error loading params from disk: {e}")
        return {}


def step_requires_metric(obj: Any) -> bool:
    sig = inspect.signature(obj.step)

    for param in sig.parameters.values():
        if param.name == "metrics":
            return param.default == inspect.Parameter.empty
    return False
