import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUNS = Path(__file__).resolve().parent


def load(tag):
    with open(RUNS / f"{tag}.pkl", "rb") as f:
        return pickle.load(f)


def report(tag):
    d = load(tag)
    print(f"\n=== {tag} (digit {d['d0']} -> digit {d['d1']}) ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    regimes = [("a", "independent init"), ("b", "shared init"), ("c", "warm-start")]

    for ax, (key, title) in zip(axes, regimes):
        rows = d[f"dist_{key}"]
        dt = np.array([r["dt"] for r in rows])
        raw = np.array([r["raw"] for r in rows])
        aligned = np.array([r["aligned"] for r in rows])
        func = np.array([r["functional"] for r in rows])

        rho_raw, _ = spearmanr(dt, raw)
        rho_aligned, _ = spearmanr(dt, aligned)
        rho_func, _ = spearmanr(dt, func)
        rho_af, _ = spearmanr(aligned, func)

        ax.scatter(dt, raw, s=14, alpha=0.5, label=f"raw (ρ={rho_raw:.2f})", color="tab:red")
        ax.scatter(dt, aligned, s=14, alpha=0.5, label=f"aligned (ρ={rho_aligned:.2f})", color="tab:blue")
        ax.set_xlabel("|Δt|  (ground-truth image distance)")
        ax.set_ylabel("weight distance")
        ax.set_title(f"regime ({key}): {title}")
        ax.legend(fontsize=8)

        print(f"  regime {key:1s} | Spearman(dt, raw weight)     = {rho_raw:+.3f}")
        print(f"          | Spearman(dt, aligned weight) = {rho_aligned:+.3f}")
        print(f"          | Spearman(dt, functional)     = {rho_func:+.3f}")
        print(f"          | Spearman(aligned, functional)= {rho_af:+.3f}  (does aligned weight dist track actual image diff?)")

    plt.tight_layout()
    out_path = RUNS / f"{tag}_distance_vs_dt.png"
    plt.savefig(out_path, dpi=130)
    print(f"  saved {out_path}")

    # basin-jump detector on regime (c) consecutive distances
    consec = d["consec_c"]
    t_c = np.array([r["t"] for r in consec])
    aligned_c = np.array([r["aligned"] for r in consec])
    med = np.median(aligned_c)
    mad = np.median(np.abs(aligned_c - med)) + 1e-8
    z = (aligned_c - med) / mad
    jump_idx = np.where(z > 3)[0]
    print(f"  basin-jump candidates (aligned dist MAD-z > 3): t ≈ {np.round(t_c[jump_idx], 3).tolist()}")

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(t_c, aligned_c, marker="o", ms=3)
    for i in jump_idx:
        ax2.axvline(t_c[i], color="red", alpha=0.4, linestyle="--")
    ax2.set_xlabel("t (along warm-started sweep)")
    ax2.set_ylabel("aligned weight dist to previous t")
    ax2.set_title(f"{tag}: basin-jump trace (regime c)")
    plt.tight_layout()
    out_path2 = RUNS / f"{tag}_basin_jumps.png"
    plt.savefig(out_path2, dpi=130)
    print(f"  saved {out_path2}")


if __name__ == "__main__":
    tags = sys.argv[1:] if len(sys.argv) > 1 else ["easy_1_1", "hard_3_8"]
    for t in tags:
        report(t)