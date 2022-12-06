import torch
from torch import Tensor, nn


def get_activation(name: str, **kwargs) -> nn.Module:
    """Returns an activation function given its name

    Args:
        name (str): Name of activation function

    Returns:
        nn.Module: Activation function
    """

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "glu":
        return GLU(**kwargs)
    if name == "swish":
        return Swish()

    raise NotImplementedError(f"Unkown activation function {name}")


class GLU(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x, x_gate = x.chunk(2, dim=self.dim)
        return x * torch.sigmoid(x_gate)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
