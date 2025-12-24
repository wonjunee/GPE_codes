#!/usr/bin/env python3
"""
Train a conditional flow matching (CFM) velocity field R in the *latent* space
after training a GPE encoder T and decoder S.

High-level:
  - T : image -> latent
  - S : latent -> image
  - R : (latent, t) -> velocity in latent

Training objective (standard flow-matching style):
  Sample x ~ data, z ~ N(0,I), t ~ Uniform[0,1]
  Tx = scale * T(x)   (or scale from a stored scalar)
  x_t = (1-t) z + t Tx
  u   = Tx - z
  Minimize E || R(x_t, t) - u ||^2

Logging every `--plot_freq`:
  - Recon preview: x vs S(T(x))
  - Generation preview: S(R_push(z))
  - FID-like score using your hadamard/sqrt formula via utilfunctions
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid, save_image
import tqdm

from utilfunctions import *


# ----------------------------
# Repro / device helpers
# ----------------------------
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda was set but CUDA is not available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device: torch.device):
    """
    weights_only=True exists in newer PyTorch. Fall back cleanly for older versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_wrap_dataparallel(model: torch.nn.Module, force: bool, device: torch.device) -> torch.nn.Module:
    """
    Wrap in DataParallel if requested and CUDA is available.
    """
    if force and device.type == "cuda":
        return torch.nn.DataParallel(model).to(device)
    return model.to(device)


@torch.inference_mode()
def collect_validation_batch(
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    non_blocking: bool,
) -> torch.Tensor:
    """
    Collect exactly n_samples images once for stable logging.
    """
    chunks = []
    n = 0
    for imgs, _ in loader:
        need = n_samples - n
        take = min(need, imgs.size(0))
        chunks.append(imgs[:take].contiguous())
        n += take
        if n >= n_samples:
            break
    x_val = torch.cat(chunks, dim=0)
    return x_val.to(device, non_blocking=non_blocking).detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--num_iter", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # dataset / device / io tags
    parser.add_argument("--data", type=str, default="celeb")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--saving", type=str, default="0")
    parser.add_argument("--fig", type=str, default="0")

    # logging / eval
    parser.add_argument("--plot_freq", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=1000)       # used for x_val previews
    parser.add_argument("--fid_real", type=int, default=1000)       # number of real samples for your FID formula
    parser.add_argument("--fid_gen", type=int, default=1000)        # z_val size
    parser.add_argument("--Nt_plot", type=int, default=10)
    parser.add_argument("--save_R_every", type=int, default=10_000)

    # parallelism
    parser.add_argument("--S_parallel", action="store_true", help="Wrap S in DataParallel")
    parser.add_argument("--T_parallel", action="store_true", help="Wrap T in DataParallel (only if your ckpt needs it)")

    # GPU visibility
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='Optional CUDA_VISIBLE_DEVICES string like "0,1,2". If omitted, do not override.',
    )

    return parser.parse_args()

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    print(args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # system preferences
    torch.set_default_dtype(torch.float32)
    seed = int(np.random.randint(100))
    seed_everything(seed)

    device = get_device(require_cuda=args.cuda)
    cuda = device.type == "cuda"
    num_gpus = torch.cuda.device_count() if cuda else 0
    print(f"Device: {device} | GPUs visible: {num_gpus}" if cuda else f"Device: {device}")

    # Dataset + config
    dataset_nn, PARAM, img_size, xshape = build_dataset_and_params(args.data)
    zdim = PARAM.zdim
    scale = PARAM.scale
    print(f"zdim: {zdim} | scale: {scale} | data: {args.data} | saving tag: {args.saving}")

    TransportT, TransportG, NetSingle = get_transport_classes(args.data)

    # data loader
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

    # output dirs
    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    print("saving images in", save_fig_dir)
    print("saving data in", save_data_dir)

    # fixed validation batch for stable visuals
    x_val = collect_validation_batch(
        loader=dataloader,
        device=device,
        n_samples=max(args.val_size, 25),
        non_blocking=non_blocking,
    )

    # fixed latent z for FID/gen previews
    z_val = torch.randn((args.fid_gen, zdim), device=device)

    # ----------------------------
    # Load pretrained T and S
    # ----------------------------
    T = TransportT(input_shape=xshape, zdim=zdim)
    T = maybe_wrap_dataparallel(T, force=args.T_parallel, device=device)
    T_ckpt = safe_torch_load(save_data_dir / "T.pth", device=device)
    scale_payload = safe_torch_load(save_data_dir / "scale.pth", device=device)

    # Support both raw and {"state_dict": ...} formats
    if isinstance(T_ckpt, dict) and "state_dict" in T_ckpt:
        T.load_state_dict(T_ckpt["state_dict"])
    else:
        T.load_state_dict(T_ckpt)
    print("Loaded encoder T.")

    # Freeze encoder and define scaling wrapper
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    @torch.inference_mode()
    def T_scale(x: torch.Tensor) -> torch.Tensor:
        return T(x) * scale_payload["C"]

    S = TransportG(output_shape=xshape, zdim=zdim)
    S = maybe_wrap_dataparallel(S, force=args.S_parallel, device=device)
    S_ckpt = safe_torch_load(save_data_dir / "S.pth", device=device)

    if isinstance(S_ckpt, dict) and "state_dict" in S_ckpt:
        S.load_state_dict(S_ckpt["state_dict"])
    else:
        S.load_state_dict(S_ckpt)
    print("Loaded decoder S.")

    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)

    # ----------------------------
    # Train R
    # ----------------------------
    print("XXX Training R XXX")
    R = NetSingle(xdim=zdim, zdim=zdim).to(device)
    optR = torch.optim.Adam(R.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # timing
    window = 1000
    t0 = time.time()
    t_last = t0

    pbar = tqdm.tqdm(total=args.num_iter)
    figure_count = 0

    R.train()
    global_iter = 0
    while global_iter < args.num_iter:
        for imgs, _ in dataloader:
            if global_iter >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=non_blocking)

            # latent targets
            with torch.inference_mode():
                Tx = T_scale(x)

            z = get_latent_samples(shape=(x.shape[0], zdim), device=device)

            # flow-matching objective
            t = torch.rand(x.shape[0], 1, device=device)
            x_t = (1.0 - t) * z + t * Tx
            u = Tx - z

            optR.zero_grad(set_to_none=True)
            loss = (R(x_t, t) - u).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), args.grad_clip)
            optR.step()

            global_iter += 1
            pbar.update(1)
            pbar.set_postfix({"iter": global_iter, "loss": f"{loss.item():.2e}"})

            # ----------------------------
            # Logging / checkpoint
            # ----------------------------
            if global_iter % args.plot_freq == 0:
                # timing report
                if global_iter % window == 0:
                    t_now = time.time()
                    dt_window = t_now - t_last
                    avg_ms_window = (dt_window / window) * 1000.0
                    avg_ms_total = ((t_now - t0) / max(global_iter, 1)) * 1000.0
                    pbar.write(
                        f"[Timing] last {window} iters: {avg_ms_window:.2f} ms/iter | overall: {avg_ms_total:.2f} ms/iter"
                    )
                    t_last = t_now

                # save R occasionally
                if args.save_R_every > 0 and (global_iter % args.save_R_every == 0):
                    torch.save(R.state_dict(), save_data_dir / f"R-{global_iter}.pth")
                    pbar.write(f"[Save] R checkpoint at iter {global_iter}")

                # evaluation visuals + fid
                R.eval()
                with torch.inference_mode():
                    show = 25
                    nrow, pad = 5, 2

                    x_show = x_val[:show]
                    Tx_show = T(x_show)
                    STx_show = S(Tx_show)

                    x_disp = to_disp(x_show)
                    STx_disp = to_disp(STx_show)

                    fid_val, SRz_vis = compute_fid_hadamard_streaming_fixed_z(
                        R,
                        S,
                        z_val,
                        real_loader=dataloader,
                        N_real=args.fid_real,
                        Nt_push=args.Nt_plot,
                        scale=scale,
                        gen_chunk=256,
                        return_preview=show,
                        device=device,
                    )
                    SRz_disp = to_disp(SRz_vis)

                    grid_x = make_grid(x_disp.to(device), nrow=nrow, padding=pad)
                    grid_STx = make_grid(STx_disp.to(device), nrow=nrow, padding=pad)
                    grid_SRz = make_grid(SRz_disp.to(device), nrow=nrow, padding=pad)

                    combined = torch.cat([grid_x, grid_STx, grid_SRz], dim=2)

                    out_base = save_fig_dir / f"G-{figure_count}_iter{global_iter:09d}_fid{fid_val:.2f}.png"
                    save_image(combined, out_base)
                    pbar.write(f"[Log] iter {global_iter} | FID {fid_val:.2f} | saved {out_base.name}")

                figure_count += 1
                R.train()

    pbar.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
