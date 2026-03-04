#!/usr/bin/env python3
"""
Train a GPE/GME-style encoder T by minimizing the GME cost on a chosen dataset.

Supported datasets:
  - mnist
  - cifar
  - celeb
  - celebahq  (as in your build_dataset_and_params)

Workflow:
  1) build training dataset (optionally augmented) and train dataloader
  2) build non-augmented dataset and validation dataloader
  3) collect a fixed validation batch x_val from the non-augmented loader
  4) (optional) load existing T checkpoint from data folder
  5) train T to minimize compute_GME_cost(T, x)
  6) every eval_every:
        - compute val loss on x_val
        - save loss curve plot
        - save T checkpoint to data folder

Assumptions:
  - compute_GME_cost, build_dataset_and_params, get_transport_classes,
    safe_torch_load, set_visible_gpus, seed_everything, get_device
    exist in utilfunctions.py
  - build_dataset_and_params(data_str, augmentation=bool) exists (you just added it)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utilfunctions import *


@torch.inference_mode()
def collect_fixed_batch(
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int,
    non_blocking: bool,
) -> torch.Tensor:
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
) -> None:
    xs = np.arange(1, len(loss_hist) + 1) * eval_every
    last_loss = loss_hist[-1]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, loss_hist, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Validation GME loss")
    ax.set_title(f"Validation loss up to iter {total_iterations}\nlast: {last_loss:.2e}")
    fig.tight_layout()
    fig.savefig(save_fig_dir / "encoder_loss.png")
    plt.close(fig)


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_checkpoint_if_exists(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    strict: bool = True,
) -> bool:
    if not ckpt_path.exists():
        print(f"[Load] checkpoint not found: {ckpt_path}")
        return False

    try:
        ckpt = safe_torch_load(ckpt_path, device=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        sd = _strip_module_prefix(sd)

        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        target.load_state_dict(sd, strict=strict)

        print(f"[Load] loaded: {ckpt_path}")
        return True
    except Exception as e:
        print(f"[Load] failed: {ckpt_path} | {type(e).__name__}: {e}")
        return False


def save_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(to_save, ckpt_path)


def build_loaders(
    *,
    data_name: str,
    batch_size: int,
    cuda: bool,
    num_gpus: int,
    augmentation: bool,
) -> tuple[DataLoader, DataLoader, object, object, int, tuple]:
    """
    Returns:
      train_loader, val_loader, PARAM, img_size, xshape
    """
    # training dataset (optional augmentation)
    train_dataset, PARAM, img_size, xshape = build_dataset_and_params(data_name, augmentation=augmentation)

    # validation dataset (no augmentation)
    val_dataset, _, _, _ = build_dataset_and_params(data_name, augmentation=False)

    num_workers = (num_gpus * 8) if cuda else 0
    pin_memory = bool(cuda)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, PARAM, img_size, xshape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # training
    p.add_argument("--num_iter", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)

    # dataset / device
    p.add_argument("--data", type=str, default="celeb")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--gpus", type=str, default=None)

    # model parallel
    p.add_argument("--T_parallel", action="store_true", help="Enable torch.nn.DataParallel for T (CUDA only).")

    # augmentation
    p.add_argument("--augmentation", action="store_true", help="Enable flip/resized-crop/blur augmentations in dataset.")

    # eval / logging
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=200)

    # output tags
    p.add_argument("--saving", type=str, default="0")
    p.add_argument("--fig", type=str, default="0")

    # loading
    p.add_argument("--load", action="store_true", help="Load existing T from data folder if present.")
    p.add_argument("--load_strict", type=int, default=1)

    return p.parse_args()


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

    # Output dirs
    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    print("saving figures in", save_fig_dir)
    print("saving data in", save_data_dir)

    # Loaders
    train_loader, val_loader, PARAM, img_size, xshape = build_loaders(
        data_name=args.data,
        batch_size=args.batch_size,
        cuda=cuda,
        num_gpus=num_gpus,
        augmentation=bool(args.augmentation),
    )
    zdim = PARAM.zdim
    scale = PARAM.scale
    print(f"zdim: {zdim} | scale: {scale} | augmentation: {bool(args.augmentation)}")

    non_blocking = bool(cuda)

    # Fixed validation batch from non-augmented loader
    x_val = collect_fixed_batch(
        val_loader,
        device=device,
        n_samples=args.val_size,
        non_blocking=non_blocking,
    )
    print("x_val shape:", tuple(x_val.shape))

    # Model
    TransportT, _, _ = get_transport_classes(args.data)
    T = TransportT(input_shape=xshape, zdim=zdim).to(device)

    if args.T_parallel:
        if device.type != "cuda":
            raise RuntimeError("--T_parallel requires CUDA")
        T = torch.nn.DataParallel(T).to(device)

    # Optional load
    ckpt_path = save_data_dir / "T.pth"
    if args.load:
        _ = load_checkpoint_if_exists(
            model=T,
            ckpt_path=ckpt_path,
            device=device,
            strict=bool(args.load_strict),
        )

    optT = torch.optim.Adam(T.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Training
    loss_hist: list[float] = []
    total_iterations = 0

    t0 = time.time()
    t_last_eval = t0

    print("Starting training...")
    T.train()

    while total_iterations < args.num_iter:
        for imgs, _ in train_loader:
            if total_iterations >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=non_blocking)

            optT.zero_grad(set_to_none=True)
            loss = compute_GME_cost(T, x)
            loss.backward()
            optT.step()

            total_iterations += 1

            if args.print_every > 0 and (total_iterations % args.print_every == 0):
                elapsed = time.time() - t0
                print(f"[Train] iter {total_iterations}/{args.num_iter} | loss {loss.item():.2e} | elapsed {elapsed/60.0:.1f} min")

            if total_iterations % args.eval_every == 0:
                t_now = time.time()
                dt = t_now - t_last_eval
                avg_ms = (dt / max(args.eval_every, 1)) * 1000.0
                avg_ms_total = ((t_now - t0) / max(total_iterations, 1)) * 1000.0
                print(f"[Timing] last {args.eval_every}: {avg_ms:.2f} ms/iter | overall: {avg_ms_total:.2f} ms/iter")

                T.eval()
                with torch.inference_mode():
                    val_loss = float(compute_GME_cost(T, x_val).item())
                loss_hist.append(val_loss)
                print(f"[Val] iter {total_iterations}: GME loss = {val_loss:.2e}")

                save_validation_plot(
                    save_fig_dir=save_fig_dir,
                    eval_every=args.eval_every,
                    total_iterations=total_iterations,
                    loss_hist=loss_hist,
                )

                save_checkpoint(T, ckpt_path)
                print(f"[Save] encoder checkpoint -> {ckpt_path}")

                T.train()
                t_last_eval = t_now

    print("Training complete.")


if __name__ == "__main__":
    main()
