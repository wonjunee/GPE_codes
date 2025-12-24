#!/usr/bin/env python3
"""
Train a GPE decoder S after a pretrained encoder T has been trained.

Workflow:
  1) Load dataset and build a training dataloader.
  2) Load pretrained encoder weights T.pth.
  3) Train decoder S to minimize reconstruction MSE:
        loss = mean ||x - S(T(x))||^2
  4) Every `--eval_every` iterations:
        - compute validation reconstruction loss on a fixed batch x_val
        - save the loss curve to .npz
        - (optionally) save a checkpoint

Assumptions:
  - TransportT / TransportG and dataset wrappers (CelebADataset, CustomCelebAHQ, etc.)
    are available in transportmodules.* as in your existing code.
  - compute_reconstruction_loss_chunked(x_val, T, S, chunk_size=...) exists
    (you showed it in utilfunctions.py).
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
import tqdm

from utilfunctions import *


# ----------------------------
# Repro / device helpers
# ----------------------------
def seed_everything(seed: int = 1) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_visible_gpus(gpu_list: str | None) -> None:
    """
    Optionally set CUDA_VISIBLE_DEVICES to a comma-separated list like "0,1,2".
    If gpu_list is None, do not override the environment.
    """
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


def get_device(force_cuda_flag: bool = False) -> torch.device:
    """
    If force_cuda_flag=True, require CUDA.
    Otherwise use CUDA if available.
    """
    if force_cuda_flag and not torch.cuda.is_available():
        raise RuntimeError("--cuda was set but CUDA is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset / model imports
# ----------------------------
@torch.inference_mode()
def collect_fixed_validation_batch(
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int,
    non_blocking: bool,
) -> torch.Tensor:
    """
    Collect a fixed validation batch of exactly n_samples images.

    This avoids variation in validation curves from sampling different images
    each time.
    """
    chunks = []
    count = 0
    for imgs, _ in dataloader:
        need = n_samples - count
        take = min(need, imgs.size(0))
        chunks.append(imgs[:take].contiguous())
        count += take
        if count >= n_samples:
            break
    x_val = torch.cat(chunks, dim=0)
    return x_val.to(device, non_blocking=non_blocking).detach()


def save_loss_curve_npz(save_data_dir: Path, xs: np.ndarray, loss_arr: np.ndarray) -> None:
    save_data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_data_dir / "recons_loss_arr.npz", xs=xs, recons_loss_arr=loss_arr)


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
    parser.add_argument("--cuda", action="store_true", help="Require CUDA; error if unavailable.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='Optional CUDA_VISIBLE_DEVICES string like "0,1,2". If omitted, do not override.',
    )

    # validation / logging
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=1000)

    # I/O tags (kept to preserve your folder naming)
    parser.add_argument("--saving", type=str, default="0")
    parser.add_argument("--fig", type=str, default="0")

    # model options
    parser.add_argument(
        "--S_parallel",
        action="store_true",
        help="Wrap decoder S in torch.nn.DataParallel.",
    )

    # checkpointing
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default="T.pth",
        help="Filename of the pretrained encoder checkpoint inside the save_data_dir.",
    )
    parser.add_argument(
        "--save_decoder_every",
        type=int,
        default=1000,
        help="If > 0, save decoder checkpoint every N iterations.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    set_visible_gpus(args.gpus)
    torch.set_default_dtype(torch.float32)
    seed_everything(1)

    device = get_device(force_cuda_flag=args.cuda)
    cuda = device.type == "cuda"
    num_gpus = torch.cuda.device_count() if cuda else 0
    print(f"Device: {device} | GPUs visible: {num_gpus}" if cuda else f"Device: {device}")

    # Dataset + config
    dataset_nn, PARAM, img_size, xshape = build_dataset_and_params(args.data)
    zdim = PARAM.zdim
    scale = PARAM.scale
    print(f"zdim: {zdim} | scale: {scale} | data: {args.data} | saving tag: {args.saving}")

    # Dataloader settings
    num_workers = (num_gpus * 8) if cuda else 0
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

    # Output dirs (preserve your naming pattern)
    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    print("saving figures in", save_fig_dir)
    print("saving data in", save_data_dir)

    # Fixed validation batch
    x_val = collect_fixed_validation_batch(
        dataloader,
        device=device,
        n_samples=args.val_size,
        non_blocking=non_blocking,
    )
    print("Validation set shape:", tuple(x_val.shape))

    # Load pretrained encoder T
    TransportT, TransportG = get_transport_classes(args.data)

    T = TransportT(input_shape=xshape, zdim=zdim).to(device)
    encoder_path = save_data_dir / args.encoder_ckpt
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    # weights_only=True is available in newer PyTorch.
    try:
        state = torch.load(encoder_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(encoder_path, map_location=device)
    T.load_state_dict(state)
    print("Encoder loaded from", encoder_path)

    # Freeze encoder: no grads, eval mode.
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    C, C_stats = compute_scalar_scale_C_from_dataloader(
        dataloader=dataloader,
        encoder=T,
        max_samples=5000,
        device=device,
    )

    print("Computed C =", C)

    scale_payload = {
        "C": float(C),
        "stats": C_stats,
        "dataset": args.data,
    }

    torch.save(scale_payload, save_data_dir / "scale.pth")
    print("Saved scale to scale.pth")

    def T_scale(x: torch.Tensor) -> torch.Tensor:
        return T(x) * scale_payload["C"]

    # Decoder S
    S = TransportG(output_shape=xshape, zdim=zdim).to(device)
    if args.S_parallel:
        # DataParallel expects the model on CUDA and will replicate across visible GPUs
        S = torch.nn.DataParallel(S).to(device)

    optS = torch.optim.Adam(S.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Tracking
    recons_loss_arr: list[float] = []
    plot_freq = int(args.eval_every)

    pbar = tqdm.tqdm(total=args.num_iter)

    # Timing
    window = plot_freq
    t0 = time.time()
    t_last = t0

    # Training loop
    S.train()
    total_iterations = 0
    while total_iterations < args.num_iter:
        for imgs, _ in dataloader:
            if total_iterations >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=non_blocking)

            # Forward through frozen encoder
            Tx = T_scale(x).detach()

            # Train decoder
            optS.zero_grad(set_to_none=True)
            STx = S(Tx)
            loss = (x - STx).pow(2).mean()
            loss.backward()
            optS.step()

            total_iterations += 1
            pbar.update(1)
            pbar.set_description(f"recons loss: {loss.item():.2e}")
            pbar.set_postfix({"iter": total_iterations})

            # Periodic validation / plotting
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

                # Validation reconstruction loss on fixed batch
                S.eval()
                with torch.inference_mode():
                    # Uses your helper from utilfunctions.py (as in your snippet).
                    val_loss = compute_reconstruction_loss_chunked(x_val, T_scale, S, chunk_size=100)
                recons_loss_arr.append(float(val_loss))

                pbar.write(f"[Val] iter {total_iterations}: recon loss = {val_loss:.2e}")

                xs = np.arange(len(recons_loss_arr)) * plot_freq
                save_loss_curve_npz(save_data_dir, xs=xs, loss_arr=np.array(recons_loss_arr))

                # Optional: save decoder checkpoint
                if args.save_decoder_every and (total_iterations % args.save_decoder_every == 0):
                    ckpt_path = save_data_dir / f"S.pth"
                    # If S is DataParallel, save the underlying module
                    to_save = S.module.state_dict() if isinstance(S, torch.nn.DataParallel) else S.state_dict()
                    torch.save(to_save, ckpt_path)
                    pbar.write(f"[Save] decoder checkpoint -> {ckpt_path}")

                S.train()
                t_last = t_now

    pbar.close()


if __name__ == "__main__":
    main()
