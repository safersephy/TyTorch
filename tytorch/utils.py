import torch
from loguru import logger
def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("using cuda")
    else:
        device = "cpu"
        logger.info("using cpu")
    return device