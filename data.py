"""Image family: morph between two digit images, parameterised by t in [0, 1].

Uses torchvision's MNIST (28x28 grayscale digits, downloads on first run).
Requires network access to the MNIST mirror — if that's unavailable in your
environment, fall back to the sklearn `load_digits` (8x8) version instead.
"""
import torch
from torchvision import datasets, transforms

_MNIST = datasets.MNIST(root="./data", train=True, download=True,
                         transform=transforms.ToTensor())
_TARGETS = _MNIST.targets  # tensor of labels, shape (60000,)


def get_digit_image(digit_class: int, idx: int = 0) -> torch.Tensor:
    """Return one example of `digit_class`, normalised to [0, 1], shape (28, 28)."""
    idxs = (_TARGETS == digit_class).nonzero().flatten()
    img, _ = _MNIST[idxs[idx].item()]  # img: (1, 28, 28), already in [0, 1] via ToTensor
    return img.squeeze(0)


def morph(I0: torch.Tensor, I1: torch.Tensor, t: float) -> torch.Tensor:
    return (1 - t) * I0 + t * I1


def image_res(img: torch.Tensor) -> int:
    return img.shape[-1]