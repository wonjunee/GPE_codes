#!/usr/bin/env python3
"""
Train a conditional flow matching (CFM) velocity field R in the *latent* space
after training a GPE encoder T and decoder S.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import gc

from utilfunctions import *


# ----------------------------
# Robust checkpoint loading for S
# ----------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Checkpoint does not contain a recognizable state_dict")

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}

def _file_stable_enough(path: Path, stable_wait_s: float = 0.20) -> bool:
    try:
        st1 = path.stat()
    except FileNotFoundError:
        return False
    if st1.st_size <= 0:
        return False

    time.sleep(stable_wait_s)

    try:
        st2 = path.stat()
    except FileNotFoundError:
        return False

    return (st1.st_size == st2.st_size) and (st1.st_mtime_ns == st2.st_mtime_ns) and (st2.st_size > 0)

def reload_S_safely(
    *,
    S: nn.Module,
    ckpt_path: Path,
    device: torch.device,
    stable_wait_s: float = 0.20,
    max_tries: int = 3,
    retry_sleep_s: float = 0.10,
    strict: bool = True,
    log_prefix: str = "",
) -> bool:
    if not ckpt_path.exists():
        print(f"{log_prefix}[S reload] missing: {ckpt_path}")
        return False

    if not _file_stable_enough(ckpt_path, stable_wait_s=stable_wait_s):
        print(f"{log_prefix}[S reload] not stable yet (likely being written): {ckpt_path.name}")
        return False

    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = _strip_module_prefix(_extract_state_dict(ckpt))
            target = S.module if isinstance(S, nn.DataParallel) else S
            target.load_state_dict(sd, strict=strict)
            target.to(device)
            target.eval()
            for p in target.parameters():
                p.requires_grad_(False)
            print(f"{log_prefix}[S reload] OK: {ckpt_path.name}")
            return True
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_s)

    print(f"{log_prefix}[S reload] FAILED: {type(last_err).__name__}: {last_err}")
    return False


# ----------------------------
# Data helpers
# ----------------------------
@torch.inference_mode()
def collect_validation_batch(loader: DataLoader, device: torch.device, n_samples: int, non_blocking: bool) -> torch.Tensor:
    chunks = []
    n = 0
    for imgs, _ in loader:
        take = min(n_samples - n, imgs.size(0))
        chunks.append(imgs[:take].contiguous())
        n += take
        if n >= n_samples:
            break
    return torch.cat(chunks, dim=0).to(device, non_blocking=non_blocking).detach()


# ----------------------------
# Args & Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_iter", type=int, default=100_000_000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--data", type=str, default="celeb")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--saving", type=str, default="0")
    p.add_argument("--fig", type=str, default="0")
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--augmentation", action="store_true")
    p.add_argument("--plot_freq", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--fid_real", type=int, default=10_000)
    p.add_argument("--fid_gen", type=int, default=10_000)
    p.add_argument("--Nt_plot", type=int, default=25)
    p.add_argument("--save_R_every", type=int, default=1_000)
    
    # parallelism
    p.add_argument("--R_parallel", action="store_true", help="Wrap R in DataParallel")
    p.add_argument("--S_parallel", action="store_true", help="Wrap S in DataParallel")
    p.add_argument("--T_parallel", action="store_true", help="Wrap T in DataParallel")

    p.add_argument("--S_reload_tries", type=int, default=3)
    p.add_argument("--S_reload_stable_wait", type=float, default=0.20)
    p.add_argument("--S_reload_retry_sleep", type=float, default=0.10)
    p.add_argument("--R_ckpt", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    set_visible_gpus(args.gpus)
    torch.set_default_dtype(torch.float32)
    seed_everything(int(np.random.randint(100)))

    device = get_device(force_cuda_flag=args.cuda)
    cuda = device.type == "cuda"
    num_gpus = torch.cuda.device_count() if cuda else 0
    print(f"Device: {device} | GPUs visible: {num_gpus}" if cuda else f"Device: {device}")

    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    dataset_train, PARAM, _, xshape = build_dataset_and_params(args.data, augmentation=bool(args.augmentation))
    dataset_val, _, _, _ = build_dataset_and_params(args.data, augmentation=False)
    zdim = PARAM.zdim

    TransportT, TransportG, NetSingle = get_transport_classes(args.data)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": (num_gpus * 4) if cuda else 0,
        "pin_memory": cuda,
    }
    dataloader = DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=False, **loader_kwargs)

    x_val = collect_validation_batch(val_loader, device, 25, cuda)
    z_val = torch.randn((args.fid_gen, zdim), device=device)

    # ----------------------------
    # Load Pretrained T & Scale Stats
    # ----------------------------
    T = TransportT(input_shape=xshape, zdim=zdim).to(device)
    T_ckpt = safe_torch_load(save_data_dir / "T.pth", device=device)
    T.load_state_dict(T_ckpt["state_dict"] if isinstance(T_ckpt, dict) and "state_dict" in T_ckpt else T_ckpt)
    T = maybe_wrap_dataparallel(T, force=args.T_parallel, device=device)
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    # Fast GPU-bound scaling
    scale_payload = safe_torch_load(save_data_dir / "scale.pth", device=device)
    mean_t = torch.as_tensor(scale_payload["mean"], dtype=torch.float32, device=device)
    C_t = torch.as_tensor(scale_payload["C"], dtype=torch.float32, device=device)

    @torch.inference_mode()
    def T_scale(x: torch.Tensor) -> torch.Tensor:
        return (T(x) - mean_t) * C_t

    # ----------------------------
    # Train R
    # ----------------------------
    print("XXX Training R XXX")
    R = NetSingle(xdim=zdim, zdim=zdim).to(device)
    
    # FIXED: Corrected the checkpoint dictionary check
    if args.R_ckpt:
        R_ckpt_path = save_data_dir / args.R_ckpt
        R_ckpt = safe_torch_load(R_ckpt_path, device=device)
        if isinstance(R_ckpt, dict) and "state_dict" in R_ckpt:
            R.load_state_dict(R_ckpt["state_dict"])
        else:
            R.load_state_dict(R_ckpt)
        print(f"Flow map R loaded from {R_ckpt_path}")

    R = maybe_wrap_dataparallel(R, force=args.R_parallel, device=device)
    
    print(f"Params in T: {sum(p.numel() for p in T.parameters()):,}")
    print(f"Params in R: {sum(p.numel() for p in R.parameters()):,}")

    optR = torch.optim.Adam(R.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    t0 = t_last_window = time.time()
    window = 1000
    figure_count = global_iter = 0

    R.train()
    while global_iter < args.num_iter:
        for imgs, _ in dataloader:
            if global_iter >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=cuda)

            with torch.no_grad():
                Tx = T_scale(x)

            z = get_latent_samples(shape=(x.shape[0], zdim), device=device)
            t = torch.rand(x.shape[0], 1, device=device)
            
            # CFM Math
            x_t = (1.0 - t) * z + t * Tx
            u = Tx - z

            optR.zero_grad(set_to_none=True)
            loss = (R(x_t, t) - u).pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(R.parameters(), args.grad_clip)
            optR.step()

            global_iter += 1

            if args.print_every > 0 and (global_iter % args.print_every == 0):
                print(f"[Train] iter {global_iter} | loss {loss.item():.2e} | elapsed {(time.time() - t0)/60.0:.1f} min")

            if global_iter % window == 0:
                t_now = time.time()
                print(f"[Timing] iter {global_iter} | last {window}: {((t_now - t_last_window)/window)*1000.0:.2f} ms/iter | overall: {((t_now - t0)/global_iter)*1000.0:.2f} ms/iter")
                t_last_window = t_now

            # Checkpointing
            if args.save_R_every > 0 and (global_iter % args.save_R_every == 0):
                target_R = R.module if isinstance(R, nn.DataParallel) else R
                torch.save(target_R.state_dict(), save_data_dir / "R.pth")
                torch.save(target_R.state_dict(), save_data_dir / f"R-{global_iter//50_000 * 50_000}.pth")
                print(f"Checkpoint saved at R-{global_iter//50_000 * 50_000}.pth")

            # Evaluation & Plotting
            if global_iter % args.plot_freq == 0:
                R.eval()
                with torch.inference_mode():

                    S = TransportG(output_shape=xshape, zdim=zdim).to(device)
                    S = maybe_wrap_dataparallel(S, force=args.S_parallel, device=device)
                    S_ckpt_path = save_data_dir / "S.pth"
                    S_ckpt = safe_torch_load(S_ckpt_path, device=device)
                    S.load_state_dict(S_ckpt["state_dict"] if isinstance(S_ckpt, dict) and "state_dict" in S_ckpt else S_ckpt)
                    S = maybe_wrap_dataparallel(S, force=args.S_parallel, device=device)
                    S.eval()

                    print(f"Params in S: {sum(p.numel() for p in S.parameters()):,}")

                    # Cleanup the checkpoint dict immediately after loading to save RAM/VRAM
                    del S_ckpt
                    
                    show = 25
                    x_show = x_val[:show]
                    Tx_show = T_scale(x_show)
                    STx_show = S(Tx_show)

                    fid_gen_val, fid_rec_val, fid_pix_val, SRz_vis = compute_fid_inception_gen_and_recon(
                        R, S, T_scale, z_val, real_loader=dataloader, N_real=args.fid_real, 
                        Nt_push=args.Nt_plot,  gen_chunk=50, return_preview=show, device=device
                    )

                    combined = torch.cat([
                        make_grid(to_disp(x_show).to(device), nrow=5, padding=2),
                        make_grid(to_disp(STx_show).to(device), nrow=5, padding=2),
                        make_grid(to_disp(SRz_vis).to(device), nrow=5, padding=2)
                    ], dim=2)

                    out_base = save_fig_dir / f"G-{figure_count}_iter{global_iter:09d}_fid_gen_{fid_gen_val:.2f}_rec_{fid_rec_val:.2f}_pix_{fid_pix_val:.2f}.png"
                    save_image(combined, out_base)
                    print(f"[Log] iter {global_iter} | rec {(x_show - STx_show).pow(2).mean():.2e} | FID gen {fid_gen_val:.2f} | FID recon {fid_rec_val:.2f} | FID pixel {fid_pix_val:.2f} | saved {out_base.name}")

                    # Corner interpolation (Assuming save_corner_interpolation_8x8 is in utilfunctions.py)
                    idx = torch.randperm(x_val.shape[0])[:4]
                    corners = x_val[idx].to(device) # Keep on device directly
                    
                    out_interp = save_fig_dir / "corner_interp_8x8.png"
                    save_corner_interpolation_8x8(
                        x_corners=corners, T_scale_fn=T_scale, S=S, R=R,
                        out_path=out_interp, device=device,
                        Nt_inv=args.Nt_plot, Nt_fwd=args.Nt_plot,
                    )

                    del S
                    del STx_show
                    del SRz_vis

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                figure_count += 1
                R.train()

    print("Training complete.")

if __name__ == "__main__":
    main()
