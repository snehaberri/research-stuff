"""Permutation alignment of two SIRENs via activation correlation matching
(Git Re-Basin style), solved layer-by-layer with the Hungarian algorithm.

For each hidden layer, we find the permutation of `model_target`'s units that
best matches `model_ref`'s units (by correlation of their activations on a
shared coordinate grid), then apply that permutation to the layer's output
rows AND the next layer's input columns, which leaves the function computed
by `model_target` unchanged while making its weights more comparable to
`model_ref`'s.
"""
import copy
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def _layer_activations(model, coords):
    """Return list of (N, hidden) activation tensors, one per SineLayer."""
    _, acts = model(coords, return_activations=True)
    return acts


def _linear_layers_in_order(model):
    """The Linear modules whose *inputs* must be re-permuted for each hidden
    layer's permutation, in order: net[0].linear, net[1].linear, ..., final."""
    return [layer.linear for layer in model.net] + [model.final]


@torch.no_grad()
def align_models(model_ref, model_target, coords):
    """Return a new state_dict for model_target, permuted to align with model_ref.
    model_target's function is unchanged; only its weight layout is permuted."""
    acts_ref = _layer_activations(model_ref, coords)
    acts_tgt = _layer_activations(model_target, coords)

    aligned = copy.deepcopy(model_target)
    input_layers = _linear_layers_in_order(aligned)  # len = n_layers + 1

    for layer_idx in range(len(model_ref.net)):
        A = acts_ref[layer_idx]   # (N, hidden)
        B = acts_tgt[layer_idx]   # (N, hidden)
        # correlation-based cost: maximize correlation <=> minimize -corr
        A_c = A - A.mean(dim=0, keepdim=True)
        B_c = B - B.mean(dim=0, keepdim=True)
        A_n = A_c / (A_c.norm(dim=0, keepdim=True) + 1e-8)
        B_n = B_c / (B_c.norm(dim=0, keepdim=True) + 1e-8)
        corr = A_n.T @ B_n  # (hidden, hidden), corr[i, j] = corr(ref unit i, target unit j)
        cost = -corr.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        # col_ind[i] = which target unit should move into ref-slot i
        perm = torch.as_tensor(col_ind, dtype=torch.long)

        out_layer = aligned.net[layer_idx].linear
        out_layer.weight.data = out_layer.weight.data[perm, :].clone()
        out_layer.bias.data = out_layer.bias.data[perm].clone()

        next_layer = input_layers[layer_idx + 1]
        next_layer.weight.data = next_layer.weight.data[:, perm].clone()

    return aligned.state_dict()