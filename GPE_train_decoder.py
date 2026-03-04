#!/usr/bin/env python3
"""
Train a GPE decoder S after a pretrained encoder T has been trained.

Workflow:
  1) Load dataset and build a training dataloader (optionally augmented).
  2) Load pretrained encoder weights T.pth.
  3) Compute a scalar scale C and define T_scale(x) = C * T(x).
  4) Train decoder S to minimize reconstruction MSE:
        loss = mean ||x - S(T_scale(x))||^2
  5) Every `--eval_every` iterations:
        - compute validation reconstruction loss on a fixed batch x_val
        - save the loss curve to .npz
        - save a 4x4 reconstruction comparison figure (input vs reconstruction)
        - (optionally) save a checkpoint

Assumptions:
  - TransportT / TransportG and dataset wrappers exist in transportmodules.* (via utilfunctions).
  - compute_reconstruction_loss_chunked(x_val, T_scale, S, chunk_size=...) exists in utilfunctions.py.
  - helper utilities exist in utilfunctions.py:
        set_visible_gpus, seed_everything, get_device, build_dataset_and_params,
        get_transport_classes, safe_torch_load, maybe_wrap_dataparallel,
        compute_scalar_scale_C_from_dataloader

Important:
  - If you enable augmentation for training, the fixed validation batch is still collected
    from a non-augmented dataset to keep validation stable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from utilfunctions import *


# ----------------------------
# Small helpers
# ----------------------------
@torch.inference_mode()
def collect_fixed_validation_batch(
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
        chunks.append(imgs[:take].contiguous())
        count += take
        if count >= n_samples:
            break
    x_val = torch.cat(chunks, dim=0)
    return x_val.to(device, non_blocking=non_blocking).detach()


def save_loss_curve_npz(save_data_dir: Path, xs: np.ndarray, loss_arr: np.ndarray) -> None:
    save_data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_data_dir / "recons_loss_arr.npz", xs=xs, recons_loss_arr=loss_arr)


def _to_vis_01(u: torch.Tensor) -> torch.Tensor:
    """
    For visualization only:
      - if looks like [-1,1], map to [0,1]
      - clamp to [0,1]
    """
    u = u.detach().float().cpu()
    umin, umax = float(u.min()), float(u.max())
    if umin >= -1.1 and umax <= 1.1:
        u = (u + 1.0) / 2.0
    return u.clamp(0.0, 1.0)


@torch.inference_mode()
def save_reconstruction_grid_4x4(
    *,
    x: torch.Tensor,            # (B,C,H,W)
    T_scale,
    S: torch.nn.Module,         # pass S.module if DataParallel
    save_path: Path,
    nrow: int = 4,
    max_items: int = 16,
) -> None:
    """
    Saves a 4x4 reconstruction comparison image.
    Layout: top = input, bottom = reconstruction.
    """
    was_training = S.training
    S.eval()

    x = x[:max_items].detach()
    Tx = T_scale(x)
    x_hat = S(Tx)

    x_vis = _to_vis_01(x)
    xhat_vis = _to_vis_01(x_hat)

    grid_in = make_grid(x_vis, nrow=nrow, padding=2)
    grid_out = make_grid(xhat_vis, nrow=nrow, padding=2)
    comp = torch.cat([grid_in, grid_out], dim=1)  # stack vertically (CHW)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(comp.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Top: input   |   Bottom: reconstruction")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    if was_training:
        S.train()


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
    p.add_argument("--cuda", action="store_true", help="Require CUDA; error if unavailable.")
    p.add_argument("--gpus", type=str, default=None)

    # augmentation
    p.add_argument(
        "--augmentation",
        action="store_true",
        help="If set, apply dataset augmentations in build_dataset_and_params(..., augmentation=True) for training.",
    )

    # validation / logging
    p.add_argument("--val_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=100)

    # I/O tags
    p.add_argument("--saving", type=str, default="0")
    p.add_argument("--fig", type=str, default="0")

    # model parallel
    p.add_argument("--T_parallel", action="store_true", help="Wrap encoder T in torch.nn.DataParallel (CUDA only).")
    p.add_argument("--S_parallel", action="store_true", help="Wrap decoder S in torch.nn.DataParallel (CUDA only).")

    # checkpointing
    p.add_argument("--encoder_ckpt", type=str, default="T.pth")
    p.add_argument("--save_decoder_every", type=int, default=1000)

    # NEW: optional load decoder checkpoint from save_data_dir
    p.add_argument(
        "--decoder_ckpt",
        type=str,
        default=None,
        help="If provided, load decoder S weights from save_data_dir / decoder_ckpt before training (e.g. S.pth).",
    )
    p.add_argument(
        "--decoder_strict",
        action="store_true",
        help="If set, load decoder checkpoint with strict=True (default is strict=False).",
    )

    # optional: stop after scatter plot
    p.add_argument("--scatter_only", action="store_true")

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

    # Dataset + config
    # Training dataset may be augmented; validation dataset must NOT be augmented.
    dataset_train, PARAM, img_size, xshape = build_dataset_and_params(
        args.data, augmentation=bool(args.augmentation)
    )
    dataset_val, _, _, _ = build_dataset_and_params(args.data, augmentation=False)

    zdim = PARAM.zdim
    print(f"zdim: {zdim} | data: {args.data} | do_augment: {bool(args.augmentation)} | saving tag: {args.saving}")

    # DataLoaders
    num_workers = (num_gpus * 8) if cuda else 0
    pin_memory = bool(cuda)
    non_blocking = bool(cuda)

    dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # Fixed validation batch (non-augmented)
    x_val = collect_fixed_validation_batch(
        dataloader=val_loader,
        device=device,
        n_samples=args.val_size,
        non_blocking=non_blocking,
    )
    print("Validation batch shape:", tuple(x_val.shape))

    # Models
    TransportT, TransportG, _ = get_transport_classes(args.data)

    # ---- Load encoder T ----
    T = TransportT(input_shape=xshape, zdim=zdim)
    T_ckpt_path = save_data_dir / args.encoder_ckpt
    T_ckpt = safe_torch_load(T_ckpt_path, device=device)

    if isinstance(T_ckpt, dict) and "state_dict" in T_ckpt:
        T.load_state_dict(T_ckpt["state_dict"])
    else:
        T.load_state_dict(T_ckpt)

    T = maybe_wrap_dataparallel(T, force=args.T_parallel, device=device)
    print(f"Loaded encoder T from {T_ckpt_path}")

    # Freeze encoder
    T.eval()
    for p_ in T.parameters():
        p_.requires_grad_(False)

    # ---- Compute scalar scale C and save ----
    C, C_stats = compute_scalar_scale_C_from_dataloader(
        dataloader=dataloader,
        encoder=T,
        max_samples=5_000,
        device=device,
    )
    print("Computed C =", C)

    scale_payload = {"C": float(C), "stats": C_stats, "dataset": args.data}
    scale_path = save_data_dir / "scale.pth"
    torch.save(scale_payload, scale_path)
    print("Saved scale to", scale_path)

    # @torch.inference_mode()
    def T_scale(x: torch.Tensor) -> torch.Tensor:
        return T(x) * scale_payload["C"]

    # ---------------------------------------------------------
    # Scatter plot: Gaussian vs C*T(x)
    # ---------------------------------------------------------
    torch.manual_seed(0)
    n_scatter = 5000

    z_data = torch.empty(n_scatter, zdim, device="cpu")
    filled = 0
    for imgs, _ in dataloader:
        x = imgs.to(device, non_blocking=non_blocking)
        with torch.no_grad():
            z = T(x) * C

        b = min(z.shape[0], n_scatter - filled)
        z_data[filled:filled + b].copy_(z[:b].detach().cpu())
        filled += b
        if filled >= n_scatter:
            break

    if filled < n_scatter:
        raise RuntimeError(f"Only filled {filled} samples, need {n_scatter}")

    z_gauss = torch.randn(n_scatter, zdim)
    z_data_2d = z_data[:, :2]
    z_gauss_2d = z_gauss[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(z_gauss_2d[:, 0], z_gauss_2d[:, 1], s=5, alpha=0.4, label="Gaussian")
    plt.scatter(z_data_2d[:, 0], z_data_2d[:, 1], s=5, alpha=0.4, label="C · T(x)")
    plt.axis("equal")
    plt.legend()
    plt.title(f"Latent scatter (zdim={zdim})")

    scatter_path = save_fig_dir / "latent_scatter_gaussian_vs_data.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved latent scatter plot to", scatter_path)

    if args.scatter_only:
        print("scatter_only set, exiting now.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Decoder S
    # ---------------------------------------------------------
    S = TransportG(output_shape=xshape, zdim=zdim).to(device)

    if args.S_parallel:
        if device.type != "cuda":
            raise RuntimeError("--S_parallel requires CUDA.")
        S = torch.nn.DataParallel(S).to(device)

    # ---- Optional: load decoder checkpoint from save_data_dir ----
    if args.decoder_ckpt is not None:
        S_ckpt_path = save_data_dir / args.decoder_ckpt
        if not S_ckpt_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {S_ckpt_path}")

        S_ckpt = safe_torch_load(S_ckpt_path, device=device)

        # Load into the underlying module (handles DataParallel cleanly)
        S_target = S.module if isinstance(S, torch.nn.DataParallel) else S

        if isinstance(S_ckpt, dict) and "state_dict" in S_ckpt:
            sd = S_ckpt["state_dict"]
        else:
            sd = S_ckpt

        missing, unexpected = S_target.load_state_dict(sd, strict=bool(args.decoder_strict))
        print(f"Loaded decoder S from {S_ckpt_path}")
        if (not args.decoder_strict) and (len(missing) > 0 or len(unexpected) > 0):
            print(f"[Decoder load] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    optS = torch.optim.Adam(S.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Tracking
    recons_loss_arr: list[float] = []

    # Timing
    t0 = time.time()
    t_last_eval = t0
    t_last_print = t0

    print("Starting training...")
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

            # Print training loss
            if args.print_every > 0 and (total_iterations % args.print_every == 0):
                t_now = time.time()
                dt = t_now - t_last_print
                ms_per_iter = (dt / max(args.print_every, 1)) * 1000.0
                elapsed_min = (t_now - t0) / 60.0
                print(
                    f"[Train] iter {total_iterations}/{args.num_iter} | "
                    f"loss {loss.item():.3e} | "
                    f"{ms_per_iter:.2f} ms/iter | "
                    f"elapsed {elapsed_min:.1f} min"
                )
                t_last_print = t_now

            # Eval / plot
            if args.eval_every > 0 and (total_iterations % args.eval_every == 0):
                t_now = time.time()
                dt_window = t_now - t_last_eval
                avg_ms_window = (dt_window / max(args.eval_every, 1)) * 1000.0
                avg_ms_total = ((t_now - t0) / max(total_iterations, 1)) * 1000.0
                print(
                    f"[Timing] iter {total_iterations} | "
                    f"last {args.eval_every} iters: {avg_ms_window:.2f} ms/iter | "
                    f"overall: {avg_ms_total:.2f} ms/iter"
                )

                # Underlying module for utils/plots
                S_base = S.module if isinstance(S, torch.nn.DataParallel) else S

                # Validation loss on fixed batch
                S.eval()
                with torch.inference_mode():
                    val_loss = compute_reconstruction_loss_chunked(
                        x_val, T_scale, S_base, chunk_size=100
                    )

                recons_loss_arr.append(float(val_loss))
                print(f"[Val] iter {total_iterations}: recon loss = {val_loss:.3e}")

                xs = np.arange(len(recons_loss_arr)) * int(args.eval_every)
                save_loss_curve_npz(save_data_dir, xs=xs, loss_arr=np.array(recons_loss_arr))
                print("[Save] loss curve ->", save_data_dir / "recons_loss_arr.npz")

                # 4x4 reconstruction grid
                grid_path = save_fig_dir / f"recons_grid_iter_{total_iterations:07d}.png"
                save_reconstruction_grid_4x4(
                    x=x,
                    T_scale=T_scale,
                    S=S_base,
                    save_path=grid_path,
                    nrow=4,
                    max_items=16,
                )
                print("[Fig] saved reconstruction grid ->", grid_path)

                # Save decoder checkpoint
                if args.save_decoder_every > 0 and (total_iterations % args.save_decoder_every == 0):
                    ckpt_path = save_data_dir / "S.pth"
                    to_save = S.module.state_dict() if isinstance(S, torch.nn.DataParallel) else S.state_dict()
                    torch.save(to_save, ckpt_path)
                    print("[Save] decoder checkpoint ->", ckpt_path)

                S.train()
                t_last_eval = t_now

    print("Training complete.")


if __name__ == "__main__":
    main()
