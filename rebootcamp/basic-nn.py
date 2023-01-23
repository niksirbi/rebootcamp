from pathlib import Path
from typing import List, Tuple, Union

# import jax.numpy as jnp
# import matplotlib.pyplot as plt
import numpy as np

# from jax import grad, jit, nn

# Specify paths
repo_path = Path("/Users/nsirmpilatze/Code/NoBlackBoxes/LastBlackBox")
box_path = repo_path / "boxes" / "learning"
data_path = box_path / "supervised" / "_data" / "complex.csv"

# Load data
data = np.genfromtxt(data_path, delimiter=",")
x, y = data.T
# add extra dimension
x = np.expand_dims(x, 1)
y = np.expand_dims(y, 1)
print(x.shape, y.shape)

# Initialize network with variable n_layers and n_neurons per layer
n_layers = 2
n_neurons = 10


def initialize_network(
    n_layers: int, n_neurons: Union[int, List[int]]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Initialize network with variable n_layers and n_neurons per layer.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    n_neurons : int or list of int
        Number of neurons per layer.
        If int, all layers have the same number of neurons.

    """

    # validate inputs
    if type(n_neurons) == int:
        n_neurons_per_layer = [n_neurons] * n_layers
    elif type(n_neurons) == list:
        if len(n_neurons) != n_layers:
            raise ValueError("n_neurons list must be of length n_layers")
        else:
            if not all(isinstance(x, int) for x in n_neurons):
                raise TypeError("all elements of n_neurons list must be int")
            n_neurons_per_layer = n_neurons
    else:
        raise TypeError("n_neurons must be int or list of int")

    # initialize weights and biases lists for each layer
    W_list = []
    B_list = []
    for i in range(n_layers):
        # initialize random weights and biases for each layer
        weights = np.random.rand(n_neurons_per_layer[i]) - 0.5
        biases = np.random.rand(n_neurons_per_layer[i]) - 0.5
        # add extra dimension
        weights = np.expand_dims(weights, 0)
        biases = np.expand_dims(biases, 0)
        # append to list
        W_list.append(weights)
        B_list.append(biases)

    return (W_list, B_list)
