#!/usr/bin/env python3
"""
Train a GPE/GME-style encoder T by minimizing the GME cost on a chosen dataset.

Supported datasets (as in your original script):
  - mnist
  - cifar
  - celeb
  - celebahq

This script:
  1) builds a dataset + dataloader
  2) trains TransportT to minimize compute_GME_cost(T, x)
  3) every `--eval_every` iterations, evaluates on a fixed held-out batch and
     saves a validation-loss plot.

Assumptions:
  - compute_GME_cost is available from utils.functions (or elsewhere on PYTHONPATH)
  - TransportT and dataset helpers (CelebADataset, CustomCelebAHQ, etc.)
    live in transportmodules.* as in your original code
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import matplotlib.pyplot as plt
import tqdm

from utilfunctions import *

# ----------------------------
# Dataset setup
# ----------------------------
@torch.inference_mode()
def collect_fixed_batch(
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int,
    non_blocking: bool,
) -> torch.Tensor:
    """
    Collect exactly n_samples images from the dataloader and move to device.

    This forms a fixed validation batch so evaluation is comparable across time.
    """
    chunks = []
    count = 0
    for imgs, _ in dataloader:
        need = n_samples - count
        take = min(need, imgs.size(0))
        chunks.append(imgs[:take])
        count += take
        if count >= n_samples:
            break
    x_val = torch.cat(chunks, dim=0)
    return x_val.to(device, non_blocking=non_blocking).detach()


def save_validation_plot(
    save_fig_dir: Path,
    eval_every: int,
    total_iterations: int,
    loss_hist: list[float],
    last_loss: float,
) -> None:
    """
    Save a log-scale plot of validation loss vs iterations.
    """
    xs = np.arange(1, len(loss_hist) + 1) * eval_every
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, loss_hist, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Validation GME loss")
    ax.set_title(f"Validation loss up to iter {total_iterations}\nlast: {last_loss:.2e}")
    fig.tight_layout()
    fig.savefig(save_fig_dir / f"validation_loss_up_to_{total_iterations}.png")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--num_iter", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    # dataset / device
    parser.add_argument("--data", type=str, default="celeb")
    parser.add_argument("--cuda", action="store_true", help="Require CUDA; error if not available.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='Optional CUDA_VISIBLE_DEVICES string like "0,1,2". If omitted, do not override.',
    )

    parser.add_argument(
        "--T_parallel",
        action="store_false",
        help="Wrap encoder T in torch.nn.DataParallel.",
    )

    # validation / logging
    parser.add_argument("--val_size", type=int, default=1000, help="Size of the fixed validation batch.")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate and plot every N iterations.")

    # naming / output
    parser.add_argument("--saving", type=str, default="0")
    parser.add_argument("--fig", type=str, default="0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    # Optional: control which GPUs are visible
    set_visible_gpus(args.gpus)

    torch.set_default_dtype(torch.float32)
    seed_everything(1)

    device = get_device(force_cuda_flag=args.cuda)
    cuda = device.type == "cuda"
    num_gpus = torch.cuda.device_count() if cuda else 0
    print(f"Device: {device} | GPUs visible: {num_gpus}" if cuda else f"Device: {device}")

    # Dataset
    dataset_nn, PARAM, img_size, xshape = build_dataset_and_params(args.data)
    zdim = PARAM.zdim
    scale = PARAM.scale
    print(f"zdim: {zdim} | scale: {scale} | data: {args.data} | saving tag: {args.saving}")

    # DataLoader settings
    num_workers = num_gpus * 8 if cuda else 0
    pin_memory = bool(cuda)
    non_blocking = bool(cuda)

    dataloader = DataLoader(
        dataset_nn,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    # Fixed validation batch
    x_val = collect_fixed_batch(
        dataloader,
        device=device,
        n_samples=args.val_size,
        non_blocking=non_blocking,
    )

    # Output dirs (match your naming pattern)
    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    print("saving figures in", save_fig_dir)
    print("saving data in", save_data_dir)

    # Model
    TransportT, _, _ = get_transport_classes(args.data)
    T = TransportT(input_shape=xshape, zdim=zdim).to(device)
    if args.T_parallel:
        # DataParallel expects the model on CUDA and will replicate across visible GPUs
        T = torch.nn.DataParallel(T).to(device)

    optT = torch.optim.Adam(T.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Training loop
    loss_hist: list[float] = []

    pbar = tqdm.tqdm(total=args.num_iter)
    total_iterations = 0

    # Timing windows
    window = args.eval_every
    t0 = time.time()
    t_last = t0

    T.train()
    while total_iterations < args.num_iter:
        for imgs, _ in dataloader:
            if total_iterations >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=non_blocking)

            optT.zero_grad(set_to_none=True)
            loss = compute_GME_cost(T, x)
            loss.backward()
            optT.step()

            pbar.update(1)
            pbar.set_postfix({"iter": total_iterations, "loss": f"{loss.item():.2e}"})
            total_iterations += 1

            # Periodic evaluation
            if total_iterations % args.eval_every == 0:
                # Timing report
                t_now = time.time()
                dt_window = t_now - t_last
                avg_ms_window = (dt_window / max(window, 1)) * 1000.0
                avg_ms_total = ((t_now - t0) / max(total_iterations, 1)) * 1000.0
                pbar.write(
                    f"[Timing] last {window} iters: {avg_ms_window:.2f} ms/iter | "
                    f"overall: {avg_ms_total:.2f} ms/iter"
                )

                # Validation loss on fixed batch
                T.eval()
                with torch.inference_mode():
                    val_loss = compute_GME_cost(T, x_val).item()
                loss_hist.append(val_loss)
                pbar.write(f"[Val] iter {total_iterations}: GME loss = {val_loss:.2e}")

                # Plot curve
                save_validation_plot(
                    save_fig_dir=save_fig_dir,
                    eval_every=args.eval_every,
                    total_iterations=total_iterations,
                    loss_hist=loss_hist,
                    last_loss=val_loss,
                )

                # save checkpoint
                ckpt_path = save_data_dir / f"T.pth"
                # If T is DataParallel, save the underlying module
                to_save = T.module.state_dict() if isinstance(T, torch.nn.DataParallel) else T.state_dict()
                torch.save(to_save, ckpt_path)
                pbar.write(f"[Save] encoder checkpoint -> {ckpt_path}")

                T.train()
                t_last = t_now

    pbar.close()


if __name__ == "__main__":
    main()
