import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import get_digit_image, image_res
from siren import coord_grid
from train import train_siren

img = get_digit_image(3)
res = image_res(img)
coords = coord_grid(res)

model, loss = train_siren(img, coords, steps=1000, lr=1e-3, hidden=64, seed=0)
print(f"final MSE loss: {loss:.6f}")

with torch.no_grad():
    pred = model(coords).reshape(res, res)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(img.numpy(), cmap="gray")
axes[0].set_title("target")
axes[1].imshow(pred.numpy(), cmap="gray")
axes[1].set_title(f"SIREN fit (loss={loss:.4f})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("sanity_check.png", dpi=120)
print("saved sanity_check.png")