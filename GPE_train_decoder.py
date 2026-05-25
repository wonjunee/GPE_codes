#!/usr/bin/env python3
"""
GPE decoder Training Script.
Trains a decoder S after a pretrained encoder T has been trained.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from utilfunctions import *


# ----------------------------
# System & IO Helpers
# ----------------------------
@torch.inference_mode()
def collect_fixed_validation_batch(
    dataloader: DataLoader, device: torch.device, n_samples: int, non_blocking: bool
) -> torch.Tensor:
    chunks = []
    count = 0
    for imgs, _ in dataloader:
        take = min(n_samples - count, imgs.size(0))
        chunks.append(imgs[:take].contiguous())
        count += take
        if count >= n_samples:
            break
    return torch.cat(chunks, dim=0).to(device, non_blocking=non_blocking).detach()


def save_loss_curve_npz(save_data_dir: Path, xs: np.ndarray, loss_arr: np.ndarray) -> None:
    save_data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_data_dir / "recons_loss_arr.npz", xs=xs, recons_loss_arr=loss_arr)


def plot_reconstruction_loss(xs: np.ndarray, loss_arr: np.ndarray, save_path: Path) -> None:
    """Generates and saves the reconstruction loss plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(xs, loss_arr, linestyle='-', color='b', label='Reconstruction Loss')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('Reconstruction Loss vs Iteration')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.yscale('log') 
    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _to_vis_01(u: torch.Tensor) -> torch.Tensor:
    """Clamps and normalizes a tensor for saving to an image."""
    u = u.detach().float()
    umin, umax = float(u.min()), float(u.max())
    if umin >= -1.1 and umax <= 1.1:
        u = (u + 1.0) / 2.0
    return u.clamp(0.0, 1.0)


@torch.inference_mode()
def save_reconstruction_grid_4x4(
    x: torch.Tensor,
    T: nn.Module,
    S: nn.Module,
    mean_t: torch.Tensor,
    C_t: torch.Tensor,
    save_path: Path,
    nrow: int = 4,
    max_items: int = 16,
) -> None:
    """Saves a comparison grid to disk purely via PyTorch (No matplotlib memory leaks)."""
    was_training = S.training
    S.eval()

    x_sub = x[:max_items]
    z = (T(x_sub) - mean_t) * C_t
    x_hat = S(z)

    comp = torch.cat([_to_vis_01(x_sub), _to_vis_01(x_hat)], dim=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(comp, save_path, nrow=nrow, padding=2)

    if was_training:
        S.train()


@torch.inference_mode()
def plot_and_free_latent_scatter(
    dataloader: DataLoader, T: nn.Module, mean_t: torch.Tensor, C_t: torch.Tensor,
    device: torch.device, zdim: int, non_blocking: bool, save_path: Path, n_scatter: int = 5000
) -> None:
    """Isolates scatter plotting to safely clear CPU/GPU memory upon return."""
    z_data = torch.empty(n_scatter, zdim, device="cpu")
    filled = 0
    for imgs, _ in dataloader:
        x = imgs.to(device, non_blocking=non_blocking)
        z = (T(x) - mean_t) * C_t
        b = min(z.shape[0], n_scatter - filled)
        z_data[filled : filled + b].copy_(z[:b].cpu())
        filled += b
        if filled >= n_scatter:
            break

    if filled < n_scatter:
        raise RuntimeError(f"Only filled {filled} samples, need {n_scatter}")

    z_gauss = torch.randn(n_scatter, zdim)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(z_gauss[:, 0], z_gauss[:, 1], s=5, alpha=0.4, label="Gaussian")
    plt.scatter(z_data[:, 0], z_data[:, 1], s=5, alpha=0.4, label="C · (T(x) - mean)")
    plt.axis("equal")
    plt.legend()
    plt.title(f"Latent scatter (zdim={zdim})")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Unified Pipeline Wrapper
# ----------------------------
class CombinedModelPipeline(nn.Module):
    """
    Fuses Encoder, Scaling, and Decoder into one module.
    This prevents DataParallel from gathering and re-scattering tensors mid-loop.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean_t: torch.Tensor, C_t: torch.Tensor):
        super().__init__()
        self.T = encoder
        self.S = decoder
        # Registering states as buffers handles multi-GPU device migration automatically
        self.register_buffer("mean_t", mean_t)
        self.register_buffer("C_t", C_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = (self.T(x) - self.mean_t) * self.C_t
        return self.S(z)

    def TST_loss(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = (self.T(x) - self.mean_t) * self.C_t
        Sz = self.S(z)
        di = Sz - x
        TSz = (self.T(Sz) - self.mean_t) * self.C_t
        return (z - TSz).pow(2).mean() + di.pow(2).mean() + di.abs().mean()

# ----------------------------
# Main Configuration
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_iter", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--data", type=str, default="celeb")
    p.add_argument("--cuda", action="store_true", help="Require CUDA")
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--augmentation", action="store_true")
    p.add_argument("--val_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--saving", type=str, default="0")
    p.add_argument("--fig", type=str, default="0")
    p.add_argument("--T_parallel", action="store_true")
    p.add_argument("--S_parallel", action="store_true")
    p.add_argument("--encoder_ckpt", type=str, default="T.pth")
    p.add_argument("--save_decoder_every", type=int, default=1000)
    p.add_argument("--decoder_ckpt", type=str, default=None)
    p.add_argument("--decoder_strict", action="store_true")
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
    print(f"Device: {device} | GPUs visible: {torch.cuda.device_count() if cuda else 0}")

    # Paths
    save_fig_dir = Path(f"fig_{args.data.lower()}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{args.data.lower()}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    dataset_train, PARAM, img_size, xshape = build_dataset_and_params(args.data, augmentation=bool(args.augmentation))
    dataset_val, _, _, _ = build_dataset_and_params(args.data, augmentation=False)
    zdim, batch_size = PARAM.zdim, PARAM.batch_size

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": (torch.cuda.device_count() * 4) if cuda else 0,
        "pin_memory": cuda,
    }
    dataloader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=False, **loader_kwargs)

    x_val = collect_fixed_validation_batch(val_loader, device, args.val_size, cuda)

    # Setup Encoder T
    TransportT, TransportG, _ = get_transport_classes(args.data)
    T = TransportT(input_shape=xshape, zdim=zdim).to(device)
    T_ckpt = safe_torch_load(save_data_dir / args.encoder_ckpt, device=device)
    T.load_state_dict(T_ckpt["state_dict"] if isinstance(T_ckpt, dict) and "state_dict" in T_ckpt else T_ckpt)
    
    # We leave T unwrapped here; it gets parallelized inside the unified pipeline wrapper later
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    # Compute & Lock Scale Stats onto GPU
    stats = compute_scalar_scale_C_from_dataloader(dataloader, T, max_samples=5000, device=device)
    torch.save({"C": stats["C"], "mean": stats["mean"], "dataset": args.data}, save_data_dir / "scale.pth")
    
    mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
    C_t = torch.tensor(stats["C"], device=device, dtype=torch.float32)

    # Scatter Plot
    plot_and_free_latent_scatter(
        dataloader, T, mean_t, C_t, device, zdim, cuda, 
        save_path=save_fig_dir / "latent_scatter_gaussian_vs_data.png"
    )
    if args.scatter_only:
        sys.exit(0)

    # Setup Decoder S
    S = TransportG(output_shape=xshape, zdim=zdim).to(device)

    if args.decoder_ckpt:
        S_ckpt_path = save_data_dir / args.decoder_ckpt
        S_ckpt = safe_torch_load(S_ckpt_path, device=device)
        sd = S_ckpt["state_dict"] if isinstance(S_ckpt, dict) and "state_dict" in S_ckpt else S_ckpt
        S.load_state_dict(sd, strict=args.decoder_strict)

    print(f"Params in T: {sum(p.numel() for p in T.parameters()):,}")
    print(f"Params in S: {sum(p.numel() for p in S.parameters()):,}")

    # Build Unified Pipeline Wrapper
    model_pipeline = CombinedModelPipeline(T, S, mean_t, C_t).to(device)
    
    # Crucial: Target ONLY the S parameters inside the pipeline wrapper
    optS = torch.optim.Adam(model_pipeline.S.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # Wrap the entire unified execution graph into DataParallel
    if args.S_parallel and cuda:
        model_pipeline = nn.DataParallel(model_pipeline)

    # Training Loop
    recons_loss_arr = []
    t0 = t_last_eval = t_last_print = time.time()
    total_iters = 0

    print("Starting training...")
    model_pipeline.train()

    # Create pointer references to original modules for plotting functions compatibility
    pure_T = model_pipeline.module.T if isinstance(model_pipeline, nn.DataParallel) else model_pipeline.T
    pure_S = model_pipeline.module.S if isinstance(model_pipeline, nn.DataParallel) else model_pipeline.S

    while total_iters < args.num_iter:
        for imgs, _ in dataloader:
            if total_iters >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=cuda)
            
            optS.zero_grad(set_to_none=True)
            # The forward pass processes T, Scaling, and S fully local to each GPU
            loss = model_pipeline.TST_loss(x)
            loss.backward()
            optS.step()

            total_iters += 1
            t_now = time.time()

            # Logging
            if args.print_every > 0 and total_iters % args.print_every == 0:
                ms_per_iter = ((t_now - t_last_print) / args.print_every) * 1000.0
                print(f"[Train] iter {total_iters}/{args.num_iter} | loss {loss.item():.3e} | {ms_per_iter:.2f} ms/iter")
                t_last_print = t_now

            # Validation & Checkpointing
            if args.eval_every > 0 and total_iters % args.eval_every == 0:
                model_pipeline.eval()
                model_pipeline.S.training = False
                with torch.inference_mode():
                    val_loss = (x_val - model_pipeline(x_val)).pow(2).mean().item()
                
                recons_loss_arr.append(val_loss)
                print(f"[Val] iter {total_iters}: recon loss = {val_loss:.3e}")

                xs = np.arange(1, len(recons_loss_arr) + 1) * args.eval_every
                save_loss_curve_npz(save_data_dir, xs, np.array(recons_loss_arr))
                plot_reconstruction_loss(xs, np.array(recons_loss_arr), save_fig_dir / "0_recons_loss.png")
                
                save_reconstruction_grid_4x4(
                    x_val, pure_T, pure_S, mean_t, C_t, 
                    save_path=save_fig_dir / f"recons_grid_iter_{total_iters//50_000*50_000:07d}.png"
                )

                if args.save_decoder_every > 0 and total_iters % args.save_decoder_every == 0:
                    torch.save(pure_S.state_dict(), save_data_dir / "S.pth")

                model_pipeline.train()
                model_pipeline.S.training = True
                t_last_eval = t_last_print = time.time() 

    print("Training complete.")

if __name__ == "__main__":
    main()
