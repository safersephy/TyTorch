import torch

def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("using cuda")
    else:
        device = "cpu"
        print("using cpu")
    return device