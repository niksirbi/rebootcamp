from pathlib import Path
from typing import Union

import torch
from matplotlib import pyplot as plt


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


# Function for plotting the latents
def plot_latents(
    latents: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    title="Latents",
):
    """
    Plot the latents.

    Parameters
    ----------
    latents : torch.Tensor
        The latents to plot.
    device : torch.device
        The device to use.
    out_dir : pathlib.Path
        The output directory where the plot will be saved.
    title : str, optional
        The title of the plot, by default "Latents".
        The plot file will be saved as <title>.png.
    """

    latents_copy = latents
    if not device == torch.device("cpu"):
        latents_copy = latents.cpu()
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    for i in range(4):
        row, col = i // 2, i % 2
        ax[row, col].imshow(latents_copy[0, i, :, :], cmap="inferno")
        ax[row, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(out_dir / f"{title}.png", dpi=128)
