#!/usr/bin/env python3
"""
Train a VAE-like model (TransportT encoder + TransportG decoder) with optional:
- Reconstruction loss (MSE in pixel space)
- KL divergence regularizer on latent Gaussian
- MDS-style loss that encourages pairwise distances in latent space to match
  pairwise distances in input space (flattened images)

Notes:
- This script sets CUDA_VISIBLE_DEVICES to all available GPUs by default.
- If you want explicit single-GPU usage, pass --parallel 0 and set --cudastr cuda:K.
- Some code paths assume that your Transport modules expose:
    TransportT(input_shape=..., zdim=...)
    TransportG(output_shape=..., zdim=...)
    Parameters(batch_size=..., sample_size=..., plot_freq=..., zdim=...)
  and that CelebADataset/CustomCelebAHQ exist where needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import tqdm

# Your project utilities (kept since you import them in the original script)
from utilfunctions import *


# ----------------------------
# Utilities
# ----------------------------
def set_all_visible_gpus() -> None:
    """
    Make all detected GPUs visible (CUDA_VISIBLE_DEVICES="0,1,2,...").
    This replicates the original behavior, but is packaged as a function.
    """
    n = torch.cuda.device_count()
    if n <= 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))


def seed_everything(seed: int = 1) -> None:
    """Seed NumPy and PyTorch RNGs for (partial) reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_distances(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix (normalized by sqrt(d)).

    Args:
        x: Tensor of shape [N, d]
        eps: small clamp to avoid sqrt(0) causing NaN gradients

    Returns:
        dist: [N, N] where dist[i,j] = ||x_i - x_j|| / sqrt(d)
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x_norm_sq = (x**2).sum(dim=1)  # [N]
    dist_sq = x_norm_sq[:, None] + x_norm_sq[None, :] - 2.0 * (x @ x.t())
    dist_sq = torch.clamp(dist_sq, min=eps)
    return torch.sqrt(dist_sq) / (x.shape[1] ** 0.5)


@torch.inference_mode()
def reconstruction_loss_batched(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
    num_recons: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean per-sample reconstruction MSE on x by micro-batching.

    Returns:
        mean_loss: scalar tensor (on CPU by default)
        recons_sample: up to num_recons reconstructed images (on CPU)
    """
    model.eval()
    n_total = x.shape[0]

    total_loss = 0.0
    total_count = 0
    recons_chunks: List[torch.Tensor] = []

    for start in range(0, n_total, batch_size):
        xb = x[start : start + batch_size].to(device, non_blocking=True)
        _, recons, _, _ = model(xb)

        # Mean squared error per sample
        per_sample = (xb - recons).pow(2).flatten(1).mean(dim=1)  # [B]
        total_loss += per_sample.sum().item()
        total_count += per_sample.numel()

        # Save a few reconstructions for visualization
        if len(recons_chunks) < num_recons:
            need = num_recons - len(recons_chunks)
            recons_chunks.append(recons[:need].detach().cpu())

    mean = total_loss / max(total_count, 1)
    recons_sample = torch.cat(recons_chunks, dim=0) if recons_chunks else torch.empty(0)
    return torch.tensor(mean), recons_sample


def denorm_0p5(x: torch.Tensor) -> torch.Tensor:
    """
    If images are normalized with mean=0.5 and std=0.5, invert that transform.
    """
    return (x * 0.5 + 0.5).clamp(0, 1)


# ----------------------------
# Model
# ----------------------------
class VanillaVAE(nn.Module):
    """
    A thin wrapper around:
      - encoder: produces a hidden feature vector of size hidden_dim
      - fc_mu, fc_logvar: project hidden features to latent mean/logvar (size zdim)
      - decoder: maps latent vector back to image

    Important: this implementation assumes your `encoder(x)` already returns
    a flattened vector of length hidden_dim.
    """

    def __init__(self, zdim: int, hidden_dim: int, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.latent_dim = zdim
        self.encoder = encoder
        self.fc_mu = nn.Linear(hidden_dim, zdim)
        self.fc_logvar = nn.Linear(hidden_dim, zdim)
        self.decoder = decoder

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)  # expected shape [B, hidden_dim]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, use_noise: bool) -> torch.Tensor:
        """
        Sample z ~ N(mu, diag(exp(logvar))) using reparameterization.
        If use_noise=False, returns mu deterministically.
        """
        if not use_noise:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        # Noise in latent only if KL is active (mirrors your original logic)
        z = self.reparameterize(mu, logvar, use_noise=True)
        recons = self.decode(z)
        return z, recons, mu, logvar

    def compute_losses(
        self,
        x: torch.Tensor,
        use_recon: bool,
        kld_wt: float,
        mds_wt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute three loss terms:
          - recon: sum of squared pixel error
          - kld: KL divergence to standard normal (scaled similarly to original)
          - mds: pairwise distance matching between mu and flattened x
        """
        _, recons, mu, logvar = self(x)

        # Reconstruction loss (pixel MSE). Original code used sum, so keep sum.
        if use_recon:
            recon_loss = torch.sum((recons - x) ** 2)
        else:
            recon_loss = torch.zeros((), device=x.device)

        # rescale_factor = number of pixels/channels per sample
        rescale_factor = int(torch.prod(torch.tensor(x.shape[1:], device=x.device)).item())

        # KL divergence term (sum form)
        if kld_wt > 0:
            # Standard VAE KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            # Your original code multiplied by (rescale_factor / zdim)
            kld = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
            kld_loss = kld * (rescale_factor / max(mu.shape[1], 1))
        else:
            kld_loss = torch.zeros((), device=x.device)

        # MDS term: match pairwise distances in latent mean to input space
        if mds_wt > 0:
            mu_dist = pairwise_distances(mu)
            x_dist = pairwise_distances(x.flatten(1))
            # Original scaling: rescale_factor * batch / (N^2 - N)
            n = x.shape[0]
            scale = rescale_factor * n / max(n * n - n, 1)
            mds_loss = torch.sum((mu_dist - x_dist) ** 2) * scale
        else:
            mds_loss = torch.zeros((), device=x.device)

        return recon_loss, kld_loss, mds_loss, recons


# ----------------------------
# Dataset setup
# ----------------------------
def collect_validation_batch(
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 5000,
    non_blocking: bool = True,
) -> torch.Tensor:
    """
    Collect a fixed number of images from a dataloader to serve as a
    lightweight "validation" set (really just a held-out batch subset).
    """
    chunks: List[torch.Tensor] = []
    count = 0
    for imgs, _ in dataloader:
        chunks.append(imgs)
        count += imgs.size(0)
        if count >= n_samples:
            break
    x_val = torch.cat(chunks, dim=0)[:n_samples]
    x_val = x_val.to(device, non_blocking=non_blocking).detach()
    return x_val


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Training loop
    parser.add_argument("--num_iter", type=int, default=10_000_000)
    parser.add_argument("--num_epochs", type=int, default=10000)  # not really used in this loop
    parser.add_argument("--starting_epoch", type=int, default=0)

    # Dataset + run naming
    parser.add_argument(
        "--data",
        type=str,
        choices=["mnist", "cifar", "celeb", "celebahq", "ffhq"],
        default="ffhq",
    )
    parser.add_argument("--modelname", type=str, default="mds")
    parser.add_argument("--trial", type=str, default="1")

    # Logging / saving
    parser.add_argument("--save_freq", type=int, default=25)  # unused, preserved
    parser.add_argument("--plot_freq", type=int, default=1000)  # use this
    parser.add_argument("--load", type=int, default=0)

    # Device
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--cudastr", type=str, default="cuda:0", help="GPU device string if not parallel.")

    # Loss toggles and weights
    parser.add_argument("--use_recon", type=int, default=1, help="0 disables recon loss.")
    parser.add_argument("--kld_wt", type=float, default=0.5)
    parser.add_argument("--mds_wt", type=float, default=1.0)

    # Plotting option
    parser.add_argument("--plt_wted", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    # Match original behavior: expose all GPUs
    set_all_visible_gpus()

    args = parse_args()

    # Construct a descriptive run name
    args.modelname = f"{args.modelname}_kld_{args.kld_wt}_mds_{args.mds_wt}_{args.trial}"
    if args.use_recon == 0:
        args.modelname += "_NO_RECON"

    print(args)
    print("cudastr:", args.cudastr)

    # Seeds and dtype
    torch.set_default_dtype(torch.float32)
    seed_everything(1)

    # Dataset
    dataset_nn, PARAM, img_size, xshape = build_dataset_and_params(args.data)
    zdim = PARAM.zdim

    # Device selection
    device = torch.device(args.cudastr if torch.cuda.is_available() else "cpu")
    cuda = (device.type == "cuda")
    print("using:", device, "cuda:", cuda)

    num_gpus = torch.cuda.device_count() if cuda else 0
    print(f"GPUs available: {num_gpus}" if cuda else "GPU not available")

    # Dataloader config
    num_workers = (num_gpus * 8) if cuda else 0
    pin_memory = bool(cuda)
    non_blocking = bool(cuda)

    dataloader = DataLoader(
        dataset_nn,
        batch_size=PARAM.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    # Lightweight "validation" set: fixed 5000 samples from training loader
    x_val = collect_validation_batch(dataloader, device, n_samples=5000, non_blocking=non_blocking)
    print("Validation set shape:", tuple(x_val.shape))

    # Optional subset loader for plotting, preserved from original (not used below)
    frac = 0.05
    indices = torch.randperm(len(dataset_nn))[: int(frac * len(dataset_nn))]
    dataset_nn_plt = Subset(dataset_nn, indices)
    _ = DataLoader(
        dataset_nn_plt,
        batch_size=PARAM.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    # Output dirs
    save_fig_path = Path(f"{args.data.lower()}_{args.modelname}_fig")
    save_data_path = Path(f"{args.data.lower()}_{args.modelname}_wts")
    save_fig_path.mkdir(parents=True, exist_ok=True)
    save_data_path.mkdir(parents=True, exist_ok=True)
    print("saving images in", save_fig_path)
    print("saving data in", save_data_path)

    # ----------------------------
    # Model setup
    # ----------------------------
    # Transport encoder output dim is zdim*10 in your original.
    # Then fc_mu/logvar maps hidden_dim=zdim*10 -> zdim.
    TransportT, TransportG = get_transport_classes(args.data)

    lr = 1e-5
    b1, b2 = 0.5, 0.999

    enc_model = TransportT(input_shape=xshape, zdim=zdim * 10).to(device)
    dec_model = TransportG(output_shape=xshape, zdim=zdim).to(device)

    model = VanillaVAE(zdim=zdim, hidden_dim=zdim * 10, encoder=enc_model, decoder=dec_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    # Optionally resume
    if args.load == 1:
        ckpt = save_data_path / "TSvae.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print("Loaded model weights from", ckpt)
        else:
            print("Checkpoint not found:", ckpt, "Starting from scratch.")

    # ----------------------------
    # Training loop
    # ----------------------------
    total_it = 0
    epoch = 0
    plot_freq = int(args.plot_freq)

    val_recon_curve: List[float] = []

    # Timing windows
    window = plot_freq  # print timing once per plot checkpoint
    t0 = time.time()
    t_last = t0

    pbar = tqdm.tqdm(total=args.num_iter)

    model.train()
    while total_it < args.num_iter:
        for imgs, _ in dataloader:
            if total_it >= args.num_iter:
                break

            # Move batch to device
            x = imgs.to(device, non_blocking=non_blocking).detach()

            # Forward + loss
            opt.zero_grad(set_to_none=True)
            recon_loss, kld_loss, mds_loss, _ = model.compute_losses(
                x,
                use_recon=(args.use_recon != 0),
                kld_wt=args.kld_wt,
                mds_wt=args.mds_wt,
            )
            loss = recon_loss + args.kld_wt * kld_loss + args.mds_wt * mds_loss

            loss.backward()
            opt.step()

            # Progress bar
            pbar.set_description(
                f"it:{total_it} "
                f"recon:{recon_loss.item():.2e} "
                f"KL:{kld_loss.item():.2e} "
                f"MDS:{mds_loss.item():.2e}"
            )
            pbar.update(1)

            # Periodic checkpoint + validation visuals
            if (total_it % plot_freq) == 0:
                # Timing diagnostics
                t_now = time.time()
                dt_window = t_now - t_last
                avg_ms_window = (dt_window / max(window, 1)) * 1000.0
                avg_ms_total = ((t_now - t0) / max(total_it + 1, 1)) * 1000.0
                tqdm.tqdm.write(
                    f"[Timing] last {window} iters: {avg_ms_window:.2f} ms/iter | "
                    f"overall: {avg_ms_total:.2f} ms/iter"
                )

                # Save checkpoint
                torch.save(model.state_dict(), save_data_path / "TSvae.pth")

                # Validation recon loss and example reconstructions
                val_loss_t, recons_sample = reconstruction_loss_batched(
                    model, x_val, device, batch_size=128, num_recons=25
                )
                val_loss = float(val_loss_t.item())
                val_recon_curve.append(val_loss)

                # Save reconstruction grid: first n originals + n reconstructions
                n = 25
                grid = make_grid(
                    torch.cat([denorm_0p5(x_val[:n].cpu()), denorm_0p5(recons_sample[:n])], dim=0),
                    nrow=5,
                )
                save_image(grid, save_fig_path / f"recons_{total_it}.png")

                # Save validation curve plot
                xs = np.arange(len(val_recon_curve)) * plot_freq
                plt.figure(figsize=(5, 4))
                plt.plot(xs, val_recon_curve)
                plt.title(f"Validation Reconstruction Loss\nloss: {val_loss:.2e}")
                plt.xlabel("Iteration")
                plt.ylabel("Mean MSE per sample")
                plt.yscale("log")
                plt.tight_layout()
                plt.savefig(save_fig_path / f"loss_{total_it}.png")
                plt.close()

                # Save curve data for later analysis
                np.savez(
                    save_data_path / "recons_loss_arr.npz",
                    xs=xs,
                    recons_loss_arr=np.array(val_recon_curve),
                )

                tqdm.tqdm.write(f"Validation recon loss @ it {total_it}: {val_loss:.2e}")

                # Reset timing window
                t_last = t_now

            total_it += 1

        epoch += 1

    pbar.close()


if __name__ == "__main__":
    main()
