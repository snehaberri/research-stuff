import copy
import numpy as np
import torch

from siren import SIREN, coord_grid


def train_siren(target_img, coords, init_state=None, steps=500, lr=1e-4,
                 hidden=64, seed=None):
    model = SIREN(hidden=hidden, seed=seed)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    target = target_img.reshape(-1, 1)
    loss = None
    for _ in range(steps):
        opt.zero_grad()
        pred = model(coords)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    return model, loss.item()


def regime_a(t_values, image_fn, coords, steps=500, hidden=64, lr=1e-4):
    """Independent random init per t."""
    results = []
    for t in t_values:
        img = image_fn(t)
        seed = int(np.random.randint(0, 1_000_000))
        model, loss = train_siren(img, coords, steps=steps, hidden=hidden, lr=lr, seed=seed)
        results.append({"t": t, "model": model, "loss": loss})
    return results


def regime_b(t_values, image_fn, coords, steps=500, hidden=64, lr=1e-4, init_seed=42):
    """Shared init, independently optimized per t."""
    shared_init = SIREN(hidden=hidden, seed=init_seed).state_dict()
    results = []
    for t in t_values:
        img = image_fn(t)
        model, loss = train_siren(img, coords, init_state=shared_init, steps=steps,
                                   hidden=hidden, lr=lr)
        results.append({"t": t, "model": model, "loss": loss})
    return results


def regime_c(t_values, image_fn, coords, first_steps=500, warm_steps=100, hidden=64, lr=1e-4):
    """Warm-started continuation: each t initialised from the previous t's converged weights."""
    results = []
    prev_state = None
    for i, t in enumerate(t_values):
        img = image_fn(t)
        steps = first_steps if prev_state is None else warm_steps
        model, loss = train_siren(img, coords, init_state=prev_state, steps=steps,
                                   hidden=hidden, lr=lr)
        prev_state = model.state_dict()
        results.append({"t": t, "model": model, "loss": loss})
    return results