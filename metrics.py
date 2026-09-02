import copy
import torch

from siren import flatten_weights
from align import align_models


def weight_dist(m1, m2):
    return (flatten_weights(m1) - flatten_weights(m2)).norm().item()


@torch.no_grad()
def functional_dist(m1, m2, coords):
    return ((m1(coords) - m2(coords)) ** 2).mean().item()


def aligned_weight_dist(m1, m2, coords):
    """Align m2 onto m1, then measure weight distance. Symmetric in spirit
    but the alignment direction matters slightly; for consistency always
    align the second argument onto the first."""
    aligned_state = align_models(m1, m2, coords)
    m2_aligned = copy.deepcopy(m2)
    m2_aligned.load_state_dict(aligned_state)
    return weight_dist(m1, m2_aligned)