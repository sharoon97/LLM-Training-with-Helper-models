"""Regenerate the training-loss panel in Figure 3 from saved checkpoints."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def load_loss_history(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "loss_history" not in checkpoint:
        raise ValueError(f"No loss_history found in {path}")
    return np.asarray(checkpoint["loss_history"], dtype=float)


def smooth_loss(y11, window_size=100):
    if len(y11) < window_size:
        return np.asarray(y11, dtype=float)
    y1_smooth = F.avg_pool1d(
        torch.tensor(y11)[None, None, :],
        kernel_size=window_size,
        stride=1,
    ).squeeze()
    return y1_smooth.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output", default="results/figures/figure_3_training_loss.pdf")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_loss = smooth_loss(load_loss_history(args.random_checkpoint, device), args.window)
    pretrained_loss = smooth_loss(load_loss_history(args.pretrained_checkpoint, device), args.window)
    plt.figure(figsize=(7, 4.5))
    plt.plot(random_loss, label="Random helper")
    plt.plot(pretrained_loss, label="Pretrained helper")
    plt.xlabel("Training batch")
    plt.ylabel("Mean next-token loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
