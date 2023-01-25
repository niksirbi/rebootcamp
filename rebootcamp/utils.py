from pathlib import Path
from typing import Union

import torch


# Check for GPU availability and set device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Found cuda. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Found MPS. Using GPU.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")
    return device


# Load a string of prompts from a txt file
def load_prompts(filepath: Union[str, Path]) -> str:
    with open(filepath, "r") as f:
        return ", ".join(f.read().splitlines())
