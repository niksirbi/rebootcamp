from pathlib import Path
from typing import List, Set, Union

import jax.numpy as jnp

# import matplotlib.pyplot as plt
import numpy as np
from jax import nn

# Specify path to data
data_dir = Path(__name__).parent / "data"
data_path = data_dir / "xy_poly.csv"

# Load data
data = np.loadtxt(data_path, delimiter=",")
x, y = data.T
# add extra dimension
x = np.expand_dims(x, 1)
y = np.expand_dims(y, 1)
print(x.shape, y.shape)


class BasicNN:
    """Basic neural network class."""

    n_layers: int
    n_neurons: Union[int, List[int]]
    activation: str

    def __init__(
        self,
        n_hidden_layers,
        n_neurons,
        activation=Set("sigmoid", "relu", "tanh"),
    ):
        """Initialize network with variable n_layers and n_neurons per layer.

        Parameters
        ----------
        n_hidden_layers : int
            Number of hidden layers
        n_neurons : int or list of int
            Number of neurons per layer.
            If int, all layers have the same number of neurons.
        activation : str, optional
            Activation function.
            Must be "sigmoid", "relu", or "tanh".
            Defaults to "sigmoid"

        """
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.activation = activation

        # initialize weights and biases lists for each layer
        self.W_list = []
        self.B_list = []
        for i in range(n_hidden_layers):
            # initialize random weights and biases for each layer
            weights = np.random.rand(n_neurons) - 0.5
            biases = np.random.rand(n_neurons) - 0.5
            # add extra dimension
            weights = np.expand_dims(weights, 0)
            biases = np.expand_dims(biases, 0)
            # append to list
            self.W_list.append(weights)
            self.B_list.append(biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the network.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Output of the network.

        """
        # apply activation function
        if self.activation == "sigmoid":
            activation = nn.sigmoid
        elif self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError("activation must be sigmoid, relu, or tanh")

        # forward pass
        for i in range(self.n_hidden_layers):
            interim = activation(jnp.dot(self.W_list[i], x) + self.B_list[i])
            # Pooling for single neuron output
            output = jnp.sum(interim)

        return output


# Initialize network
n_layers = 2
n_neurons = 10
net = BasicNN(n_layers, n_neurons, activation="sigmoid")
