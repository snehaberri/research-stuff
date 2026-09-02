"""SIREN: sinusoidal representation network for fitting a single 2D image."""
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_f
            else:
                bound = np.sqrt(6.0 / in_f) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def pre_activation(self, x):
        return self.omega_0 * self.linear(x)


class SIREN(nn.Module):
    """Coordinate -> intensity MLP. hidden/n_layers fixed across all regimes
    so weight vectors from different runs are directly comparable (same shape)."""

    def __init__(self, in_f=2, hidden=64, out_f=1, n_layers=3, omega_0=30.0, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.hidden = hidden
        self.n_layers = n_layers
        layers = [SineLayer(in_f, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden, hidden, omega_0=omega_0))
        self.net = nn.ModuleList(layers)
        self.final = nn.Linear(hidden, out_f)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.uniform_(-bound, bound)

    def forward(self, x, return_activations=False):
        acts = []
        h = x
        for layer in self.net:
            h = layer(h)
            if return_activations:
                acts.append(h)
        out = self.final(h)
        if return_activations:
            return out, acts
        return out


def coord_grid(res=28):
    """Flattened (res*res, 2) grid of coordinates in [-1, 1]."""
    xs = torch.linspace(-1, 1, res)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


def flatten_weights(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])