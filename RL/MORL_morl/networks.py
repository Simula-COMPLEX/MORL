from typing import List, Type, Iterable

import numpy as np
import torch
from torch import nn


def mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    drop_rate: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Create a multi layer perceptron (MLP), which is a collection of fully-connected layers each followed by an activation function.

    Args:
        input_dim: Dimension of the input vector
        output_dim: Dimension of the output vector
        net_arch: Architecture of the neural net. It represents the number of units per layer. The length of this list is the number of layers.
        activation_fn: The activation function to use after each layer.
        drop_rate: Dropout rate
        layer_norm: Whether to use layer normalization
    """
    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0])]
    # if drop_rate > 0.0:
    #     modules.append(nn.Dropout(p=drop_rate))
    if layer_norm:
        modules.append(nn.LayerNorm(net_arch[0]))
    modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if idx == len(net_arch) - 2 and drop_rate > 0.0:
            modules.append(nn.Dropout(p=drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1]
        modules.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*modules)


class NatureCNN(nn.Module):
    """CNN from DQN nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533."""

    def __init__(self, observation_shape: np.ndarray, features_dim: int = 512):
        """CNN from DQN Nature.

        Args:
            observation_shape: Shape of the observation.
            features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = 1 if len(observation_shape) == 2 else observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(np.zeros(observation_shape)[np.newaxis]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Predicts the features from the observations.

        Args:
            observations: current observations
        """
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
        return self.linear(self.cnn(observations / 255.0))


@torch.no_grad()
def layer_init(layer, method="orthogonal", weight_gain: float = 1, bias_const: float = 0) -> None:
    """Initialize a layer with the given method.

    Args:
        layer: The layer to initialize.
        method: The initialization method to use.
        weight_gain: The gain for the weights.
        bias_const: The constant for the bias.
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        torch.nn.init.constant_(layer.bias, bias_const)


@torch.no_grad()
def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """Polyak averaging for target network parameters.

    Args:
        params: The parameters to update.
        target_params: The target parameters.
        tau: The polyak averaging coefficient (usually small).
    """
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def huber(x, min_priority=0.01):
    """Huber loss function.

    Args:
        x: The input tensor.
        min_priority: The minimum priority.

    Returns:
        The huber loss.
    """
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()
