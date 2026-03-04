#!/usr/bin/env python3
"""
Train a conditional flow matching (CFM) velocity field R in the *latent* space
after training a GPE encoder T and decoder S.

Requested changes:
  (1) Add augmentation argument: --do_augment
      - Training dataloader can be augmented.
      - Validation batch x_val is collected from a non-augmented dataset for stability.
  (2) Clean up structure
      - Remove tqdm
      - Use a simple for imgs, _ in dataloader loop (no next(...) iteration)
  (3) Reload S at every plotting step (S may be trained in another process)
      - Avoid crashes if S.pth is being written concurrently:
          * check file stability (size+mtime)
          * load on CPU first
          * retry a few times on failure
          * if reload fails, keep existing S and continue

Assumptions:
  - utilfunctions provides:
      build_dataset_and_params, get_transport_classes, safe_torch_load,
      maybe_wrap_dataparallel, seed_everything, get_device, get_latent_samples,
      compute_fid_hadamard_streaming_fixed_z, to_disp, set_visible_gpus
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utilfunctions import *


# ----------------------------
# Robust checkpoint loading for S
# ----------------------------
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Accept:
      - raw state_dict (dict[str, Tensor])
      - {"state_dict": state_dict}
    """
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # If it already looks like a state_dict
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError("Checkpoint does not contain a recognizable state_dict")


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    If keys are 'module.xxx', strip 'module.'.
    Helpful when the saving program used DataParallel and this program did not (or vice versa).
    """
    if not sd:
        return sd
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def _file_stable_enough(path: Path, stable_wait_s: float = 0.20) -> bool:
    """
    Heuristic: check that (size, mtime_ns) are unchanged after stable_wait_s.
    Reduces probability of reading a partially-written checkpoint.
    """
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
    S: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    stable_wait_s: float = 0.20,
    max_tries: int = 3,
    retry_sleep_s: float = 0.10,
    strict: bool = True,
    log_prefix: str = "",
) -> bool:
    """
    Reload decoder weights from ckpt_path without crashing if another process is writing the file.

    Returns True if reload succeeded, False otherwise.
    """
    if not ckpt_path.exists():
        print(f"{log_prefix}[S reload] missing: {ckpt_path}")
        return False

    if not _file_stable_enough(ckpt_path, stable_wait_s=stable_wait_s):
        print(f"{log_prefix}[S reload] not stable yet (likely being written): {ckpt_path.name}")
        return False

    last_err: Optional[Exception] = None

    # Always load on CPU first, then move model to device.
    for k in range(1, max_tries + 1):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = _strip_module_prefix(_extract_state_dict(ckpt))

            target = S.module if isinstance(S, torch.nn.DataParallel) else S
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


# ----------------------------
# Args
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # training
    p.add_argument("--num_iter", type=int, default=100_000_000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # dataset / device / io tags
    p.add_argument("--data", type=str, default="celeb")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--saving", type=str, default="0")
    p.add_argument("--fig", type=str, default="0")
    p.add_argument("--gpus", type=str, default=None)

    # augmentation
    p.add_argument(
        "--augmentation",
        action="store_true",
        help="If set, apply dataset augmentations in build_dataset_and_params(..., augmentation=True) for training.",
    )

    # logging / eval
    p.add_argument("--plot_freq", type=int, default=1000)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--fid_real", type=int, default=1000)
    p.add_argument("--fid_gen", type=int, default=1000)
    p.add_argument("--Nt_plot", type=int, default=10)
    p.add_argument("--save_R_every", type=int, default=1_000)

    # parallelism
    p.add_argument("--S_parallel", action="store_true", help="Wrap S in DataParallel")
    p.add_argument("--T_parallel", action="store_true", help="Wrap T in DataParallel (only if your ckpt needs it)")

    # safe reload tuning
    p.add_argument("--S_reload_tries", type=int, default=3)
    p.add_argument("--S_reload_stable_wait", type=float, default=0.20)
    p.add_argument("--S_reload_retry_sleep", type=float, default=0.10)

    p.add_argument("--R_ckpt", type=str, default=None)

    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
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

    # Output dirs
    data_str = args.data.lower()
    save_fig_dir = Path(f"fig_{data_str}_{args.saving}_{args.fig}")
    save_data_dir = Path(f"data_{data_str}_{args.saving}")
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    save_data_dir.mkdir(parents=True, exist_ok=True)
    print("saving images in", save_fig_dir)
    print("saving data in", save_data_dir)

    # Dataset:
    # - training dataset optionally augmented
    # - validation dataset never augmented (so the preview batch is stable)
    dataset_train, PARAM, img_size, xshape = build_dataset_and_params(args.data, augmentation=bool(args.augmentation))
    dataset_val, _, _, _ = build_dataset_and_params(args.data, augmentation=False)

    zdim = PARAM.zdim
    scale = PARAM.scale
    print(f"zdim: {zdim} | scale: {scale} | data: {args.data} | do_augment: {bool(args.augmentation)}")

    TransportT, TransportG, NetSingle = get_transport_classes(args.data)

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

    # Fixed validation batch for visuals (non-augmented)
    x_val = collect_validation_batch(
        loader=val_loader,
        device=device,
        n_samples=max(args.val_size, 25),
        non_blocking=non_blocking,
    )
    print("x_val shape:", tuple(x_val.shape))

    # Fixed latent z for FID/gen previews
    z_val = torch.randn((args.fid_gen, zdim), device=device)

    # ----------------------------
    # Load pretrained T and scale
    # ----------------------------
    T = TransportT(input_shape=xshape, zdim=zdim)
    T_ckpt = safe_torch_load(save_data_dir / "T.pth", device=device)
    scale_payload = safe_torch_load(save_data_dir / "scale.pth", device=device)

    if isinstance(T_ckpt, dict) and "state_dict" in T_ckpt:
        T.load_state_dict(T_ckpt["state_dict"])
    else:
        T.load_state_dict(T_ckpt)

    T = maybe_wrap_dataparallel(T, force=args.T_parallel, device=device)
    print("Loaded encoder T.")

    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    @torch.inference_mode()
    def T_scale(x: torch.Tensor) -> torch.Tensor:
        return T(x) * scale_payload["C"]

    # ----------------------------
    # Decoder S (reload at plot time)
    # ----------------------------
    S = TransportG(output_shape=xshape, zdim=zdim)
    S = maybe_wrap_dataparallel(S, force=args.S_parallel, device=device)
    S_ckpt_path = save_data_dir / "S.pth"

    # Initial load (best effort)
    ok0 = reload_S_safely(
        S=S,
        ckpt_path=S_ckpt_path,
        device=device,
        stable_wait_s=float(args.S_reload_stable_wait),
        max_tries=int(args.S_reload_tries),
        retry_sleep_s=float(args.S_reload_retry_sleep),
        strict=True,
        log_prefix="[init] ",
    )
    if not ok0:
        print("[init] WARNING: could not load S.pth safely; continuing (plots may be off until it succeeds).")

    # ----------------------------
    # Train R
    # ----------------------------
    print("XXX Training R XXX")
    R = NetSingle(xdim=zdim, zdim=zdim).to(device)

    if args.R_ckpt is not None:
        R_ckpt_path = save_data_dir / args.R_ckpt
        R_ckpt = safe_torch_load(R_ckpt_path, device=device)
        if isinstance(T_ckpt, dict) and "state_dict" in R_ckpt:
            R.load_state_dict(R_ckpt["state_dict"])
        else:
            R.load_state_dict(R_ckpt)

    optR = torch.optim.Adam(R.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    t0 = time.time()
    t_last_window = t0
    window = 1000

    figure_count = 0
    global_iter = 0

    R.train()
    while global_iter < args.num_iter:
        for imgs, _ in dataloader:
            if global_iter >= args.num_iter:
                break

            x = imgs.to(device, non_blocking=non_blocking)

            with torch.inference_mode():
                Tx = T_scale(x)

            z = get_latent_samples(shape=(x.shape[0], zdim), device=device)

            t = torch.rand(x.shape[0], 1, device=device)
            x_t = (1.0 - t) * z + t * Tx
            u = Tx - z

            optR.zero_grad(set_to_none=True)
            loss = (R(x_t, t) - u).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), args.grad_clip)
            optR.step()

            global_iter += 1

            # periodic training print
            if args.print_every > 0 and (global_iter % args.print_every == 0):
                elapsed = time.time() - t0
                print(f"[Train] iter {global_iter} | loss {loss.item():.2e} | elapsed {elapsed/60.0:.1f} min")

            # timing report
            if global_iter % window == 0:
                t_now = time.time()
                dt_window = t_now - t_last_window
                avg_ms_window = (dt_window / window) * 1000.0
                avg_ms_total = ((t_now - t0) / max(global_iter, 1)) * 1000.0
                print(
                    f"[Timing] iter {global_iter} | last {window}: {avg_ms_window:.2f} ms/iter | "
                    f"overall: {avg_ms_total:.2f} ms/iter"
                )
                t_last_window = t_now

            # ----------------------------
            # Logging / checkpoint
            # ----------------------------
            if global_iter % args.plot_freq == 0:
                # save R occasionally
                if args.save_R_every > 0 and (global_iter % args.save_R_every == 0):
                    ckpt_R = save_data_dir / f"R-{global_iter}.pth"
                    torch.save(R.state_dict(), ckpt_R)
                    print(f"[Save] R checkpoint -> {ckpt_R.name}")

                # reload S safely (best effort)
                _ = reload_S_safely(
                    S=S,
                    ckpt_path=S_ckpt_path,
                    device=device,
                    stable_wait_s=float(args.S_reload_stable_wait),
                    max_tries=int(args.S_reload_tries),
                    retry_sleep_s=float(args.S_reload_retry_sleep),
                    strict=True,
                    log_prefix=f"[iter {global_iter}] ",
                )

                # evaluation visuals + fid
                R.eval()
                with torch.inference_mode():
                    show = 25
                    nrow, pad = 5, 2

                    x_show = x_val[:show]
                    Tx_show = T_scale(x_show)
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

                    print(f"[Log] iter {global_iter} | FID {fid_val:.2f} | saved {out_base.name}")

                figure_count += 1
                R.train()

    print("Training complete.")


if __name__ == "__main__":
    main()
