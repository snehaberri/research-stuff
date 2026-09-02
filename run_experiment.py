"""Step 2-4: run regimes (a) independent, (b) shared-init, (c) warm-start on
a digit-morph pair, compute raw/aligned weight distance + functional distance
vs |delta t|, and save checkpoints + a results dict to disk.

Usage: python run_experiment.py --d0 3 --d1 8 --n 20 --tag hard_3_8
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from data import get_digit_image, morph, image_res
from siren import coord_grid
from train import regime_a, regime_b, regime_c
from metrics import weight_dist, functional_dist, aligned_weight_dist

OUT = Path("/Users/snehasinha/INRs/research-stuff")
OUT.mkdir(exist_ok=True)


def run(d0, d1, n_t, tag, hidden=64, steps=500, warm_steps=100, lr=1e-3, n_align_pairs=40,
        idx0=0, idx1=1):
    I0, I1 = get_digit_image(d0, idx0), get_digit_image(d1, idx1)
    res = image_res(I0)
    coords = coord_grid(res)
    t_values = np.linspace(0, 1, n_t)

    def image_fn(t):
        return morph(I0, I1, t)

    print(f"[{tag}] training regime (a) independent init x{n_t} ...")
    res_a = regime_a(t_values, image_fn, coords, steps=steps, hidden=hidden, lr=lr)
    print(f"[{tag}] training regime (b) shared init x{n_t} ...")
    res_b = regime_b(t_values, image_fn, coords, steps=steps, hidden=hidden, lr=lr)
    print(f"[{tag}] training regime (c) warm-start x{n_t} ...")
    res_c = regime_c(t_values, image_fn, coords, first_steps=steps, warm_steps=warm_steps,
                      hidden=hidden, lr=lr)

    fit_losses = {
        "a": [r["loss"] for r in res_a],
        "b": [r["loss"] for r in res_b],
        "c": [r["loss"] for r in res_c],
    }
    print(f"[{tag}] mean fit MSE  a={np.mean(fit_losses['a']):.2e}"
          f"  b={np.mean(fit_losses['b']):.2e}  c={np.mean(fit_losses['c']):.2e}")

    def pairwise_distances(results, n_pairs, do_align=True):
        """Sample random pairs (i, j) and compute |dt|, raw dist, aligned dist, functional dist."""
        n = len(results)
        rng = np.random.default_rng(0)
        idx_pairs = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(n_pairs)]
        rows = []
        for i, j in idx_pairs:
            mi, mj = results[i]["model"], results[j]["model"]
            ti, tj = results[i]["t"], results[j]["t"]
            dt = abs(ti - tj)
            raw = weight_dist(mi, mj)
            fdist = functional_dist(mi, mj, coords)
            row = {"dt": dt, "raw": raw, "functional": fdist}
            if do_align:
                row["aligned"] = aligned_weight_dist(mi, mj, coords)
            rows.append(row)
        return rows

    print(f"[{tag}] computing pairwise distances (raw + aligned + functional) ...")
    dist_a = pairwise_distances(res_a, n_align_pairs)
    dist_b = pairwise_distances(res_b, n_align_pairs)
    dist_c = pairwise_distances(res_c, n_align_pairs)

    # consecutive-t distances for regime c (for the basin-jump detector)
    consec_c = []
    for i in range(len(res_c) - 1):
        mi, mj = res_c[i]["model"], res_c[i + 1]["model"]
        dt = abs(res_c[i]["t"] - res_c[i + 1]["t"])
        consec_c.append({
            "t": res_c[i]["t"],
            "dt": dt,
            "raw": weight_dist(mi, mj),
            "aligned": aligned_weight_dist(mi, mj, coords),
            "functional": functional_dist(mi, mj, coords),
        })

    out = {
        "tag": tag, "d0": d0, "d1": d1, "hidden": hidden, "res": res,
        "t_values": t_values, "fit_losses": fit_losses,
        "dist_a": dist_a, "dist_b": dist_b, "dist_c": dist_c,
        "consec_c": consec_c,
    }
    with open(OUT / f"{tag}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[{tag}] saved results to {OUT / f'{tag}.pkl'}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d0", type=int, default=1)
    p.add_argument("--d1", type=int, default=1)
    p.add_argument("--n", type=int, default=20, help="number of t values")
    p.add_argument("--tag", type=str, default="easy_1_1")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--idx0", type=int, default=0)
    p.add_argument("--idx1", type=int, default=1)
    args = p.parse_args()
    run(args.d0, args.d1, args.n, args.tag, hidden=args.hidden, steps=args.steps,
        idx0=args.idx0, idx1=args.idx1)