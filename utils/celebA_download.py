# celebA_autoencoder.py
# ------------------------------------------------------------
# 1) Download CelebA via Hugging Face and save to disk
# 2) Build PyTorch Dataset/DataLoader from saved images
# 3) Train a simple conv autoencoder (unsupervised)
# ------------------------------------------------------------
import os
from pathlib import Path
from functools import partial

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob

# ---------- Configuration ----------
SAVE_ROOT = Path("./data/celebA")  # <-- change me
IMG_SIZE = 64                        # resize for quick training
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 2                           # bump for better quality
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------

SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# ============ 1) Download & Save ============

def _save_split(hf_split, out_dir: Path):
    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(tqdm(hf_split, desc=f"Saving to {out_dir.name}")):
        # filename
        fname = f"{i:06d}.jpg"
        fpath = out_img / fname
        # save image
        img = ex["image"]  # PIL.Image
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(fpath, quality=95)
        # (optional) store metadata
        rows.append({
            "index": i,
            "filename": fname,
            "identity": ex.get("identity", None),
        })
    # save a simple index CSV (keep more columns if you like)
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)


def ensure_download(save_root: Path):
    # If we already have files, skip
    train_dir = save_root / "train"
    val_dir = save_root / "valid"
    test_dir = save_root / "test"
    have_all = (
        (train_dir / "images").exists()
        and any((train_dir / "images").iterdir())
        and (val_dir / "images").exists()
        and any((val_dir / "images").iterdir())
        and (test_dir / "images").exists()
        and any((test_dir / "images").iterdir())
    )

    if have_all and (val_dir / "images").exists() and (test_dir / "images").exists():
        print("[Info] Found existing CelebA folders. Skipping download.")
        return

    print("[Info] Downloading CelebA from Hugging Face…")
    ds = load_dataset("eurecom-ds/celeba")  # works over SSH; caches under ~/.cache

    # Map HF splits to our folder names
    split_map = {
        "train": save_root / "train",
        "validation": save_root / "valid",
        "test": save_root / "test",
    }
    for split_name, out_dir in split_map.items():
        _save_split(ds[split_name], out_dir)


ensure_download(SAVE_ROOT)

# ============ 2) PyTorch Dataset ============

class ImageFolderFlat(Dataset):
    """
    Minimal dataset that reads all jpg/png under root/images (no labels).
    """
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(
            glob(str(self.root / "images" / "*.jpg"))
            + glob(str(self.root / "images" / "*.png"))
        )
        if not self.files:
            raise RuntimeError(f"No images found under {self.root}/images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img  # unsupervised: no label

common_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

train_set = ImageFolderFlat(SAVE_ROOT / "train", transform=common_tf)
val_set   = ImageFolderFlat(SAVE_ROOT / "valid", transform=common_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"[Info] Train images: {len(train_set):,} | Val images: {len(val_set):,}")

# ============ 3) Simple Conv Autoencoder ============

class ConvAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder: 64x64 -> 4x4
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),      # 32x32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),    # 16x16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),   # 8x8
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(True),   # 4x4
        )
        self.enc_fc = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder: 4x4 <- latent
        self.dec_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # 32x32
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),         # 64x64
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 512, 4, 4)
        xrec = self.dec(h)
        return xrec

    def forward(self, x):
        z = self.encode(x)
        xrec = self.decode(z)
        return xrec, z

model = ConvAE(latent_dim=128).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()  # L1 is robust for reconstructions

def denorm(x):
    # inverse of Normalize(mean=.5, std=.5)
    return (x * 0.5 + 0.5).clamp(0, 1)

# Training loop (concise)
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for imgs in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        recons, _ = model(imgs)
        loss = criterion(recons, imgs)
        loss.backward()
        opt.step()
        running += loss.item() * imgs.size(0)
    print(f"  Train L1: {running / len(train_loader.dataset):.4f}")

    # quick val
    model.eval()
    with torch.no_grad():
        imgs = next(iter(val_loader)).to(DEVICE)
        recons, _ = model(imgs)
        val_loss = criterion(recons, imgs).item()
        print(f"  Val L1 (1 batch): {val_loss:.4f}")

        # save a small grid of reconstructions
        from torchvision.utils import save_image, make_grid
        grid = make_grid(torch.cat([denorm(imgs[:8]), denorm(recons[:8])], dim=0), nrow=8)
        (SAVE_ROOT / "samples").mkdir(exist_ok=True, parents=True)
        out_path = SAVE_ROOT / "samples" / f"recons_epoch{epoch}.png"
        save_image(grid, out_path)
        print(f"  Wrote {out_path}")

print("[Done] Trained a tiny conv autoencoder on CelebA.")
