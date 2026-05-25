from typing import Tuple
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from natsort import natsorted
from pathlib import Path
from glob import glob
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets
import scipy.linalg
import pickle


from PIL import Image, ImageFilter, ImageOps

def get_latent_samples(shape, device):
    return torch.randn(shape, device=device)
    
def swap_axes_images(img):
    img = np.transpose(img, (1,2,0))
    return img

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, sigma=0.1, n_mode=8, dim=20, size=30_000):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): transform to be applied to each image sample
        """
        self.sigma = sigma
        self.size = size
        self.centers = []
        # self.centers = [np.zeros((dim))]
        for i in range(n_mode):
            x = np.cos(2.0*np.pi*i/n_mode)
            y = np.sin(2.0*np.pi*i/n_mode)
            point = np.random.randn(dim) * 0.5

            point[0] = x * 1.5
            point[1] = y * 1.5
            self.centers.append(point)
            
        n_mode2 = 2*n_mode
        for i in range(n_mode2):
            x = np.cos(2.0*np.pi*(i+0.5)/n_mode2)
            y = np.sin(2.0*np.pi*(i+0.5)/n_mode2)
            point = np.random.randn(dim) * 0.2

            point[0] = x * 3.0
            point[1] = y * 3.0
            self.centers.append(point)

    def __len__(self): 
        return self.size

    def __getitem__(self, idx):
        point = np.random.randn(self.dim)*self.sigma
        index = np.random.randint(0, len(self.centers)-1)
        center = self.centers[index]
        for d in range(self.dim):
            point[d] += center[d]
        return point, index
        
  
class CelebADataset(torch.utils.data.Dataset):
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
        return img, 0  # unsupervised: no label

class FFHQDataset(torch.utils.data.Dataset):
    """
    Minimal dataset that reads all jpg/png under root/images (no labels).
    """
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(
            glob(str(self.root / "*.jpg"))
            + glob(str(self.root / "*.png"))
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
        return img, 0  # unsupervised: no label
  
class CustomCelebAHQ(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            data (numpy array or tensor): The data to be used in the dataset.
            labels (numpy array or tensor): The labels corresponding to the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0 # the second one is nothing. This is just for the consistency with the other dataset


import torch

@torch.no_grad()
def pushforward(net, x0, Nt=10, eps=1e-4):
    """
    Forward integration (e.g., noise to data).
    Uses an epsilon buffer to avoid t=0 and t=1 singularities.
    """
    # Create a safe time grid from eps to 1.0 - eps
    t_steps = torch.linspace(eps, 1.0 - eps, Nt + 1)
    
    for i in range(Nt):
        t0 = t_steps[i].item()
        t1 = t_steps[i+1].item()
        x0 = odeint2(net, x0, t0, t1) # Assuming odeint2 is your solver
        
    return x0

@torch.no_grad()
def pushforward_inv(net, x0, Nt=10, eps=1e-4):
    """
    Backward integration (e.g., data to noise).
    Uses an epsilon buffer to avoid t=1 and t=0 singularities.
    """
    # Create a safe time grid stepping backwards from 1.0 - eps to eps
    t_steps = torch.linspace(1.0 - eps, eps, Nt + 1)
    
    for i in range(Nt):
        t0 = t_steps[i].item()
        t1 = t_steps[i+1].item()
        x0 = odeint2(net, x0, t0, t1)
        
    return x0

def odeint1(odefun, z, t0, t1):
    """
        Forward Euler integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param t0:     float, starting time
    :param t1:     float, end time
    """
    
    h = t1 - t0 # step size
    z0 = z
    z = z0 + h * odefun(z0, t = t0)
    return z        

def odeint2(odefun, z, t0, t1): # Heun
    """
        Heun's method (Runge-Kutta 2nd order)
    """
    h = t1 - t0
    
    # Step 1: Compute velocity at current state (1st eval)
    k1 = odefun(z, t=t0)
    
    # Step 2: Estimate intermediate state
    z_intermediate = z + h * k1
    
    # Step 3: Compute velocity at intermediate state (2nd eval)
    k2 = odefun(z_intermediate, t=t1)
    
    # Final step: Average the velocities
    return z + (h / 2.0) * (k1 + k2)

def odeint4(odefun, z, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param t0:     float, starting time
    :param t1:     float, end time
    """
    
    # return z + (t1-t0) * odefun(z.detach(), t=t0)

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t = t0)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t = t0+(h/2) )
    z = z + (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t = t0+(h/2) )
    z = z + (2.0/6.0) * K

    K = h * odefun( z0 + K , t = t0+h )
    z = z + (1.0/6.0) * K

    return z      

def cdist_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (n, d), b: (m, d)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = (a * a).sum(dim=1, keepdim=True)          # (n, 1)
    b2 = (b * b).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
    d2 = a2 + b2 - 2.0 * (a @ b.transpose(0, 1))   # (n, m)
    return d2.clamp_min(0.0)                       # numerical safety

def compute_GME_cost(T, x, more: bool = False):
    # Flatten once
    Tx = T(x).view(x.shape[0], -1)
    x_flat = x.view(x.shape[0], -1)

    # Pairwise squared distances (no sqrt)
    Txy = cdist_squared(Tx, Tx)
    xy  = cdist_squared(x_flat, x_flat)

    # Log(1 + distance^2) as before
    ATxy = (1.0 + Txy).log()
    Axy  = (1.0 + xy ).log()

    loss = ((ATxy - Axy) ** 2).mean()

    if more:
        # Grab upper-triangular (i < j) entries and convert to Euclidean distances for output
        n = x_flat.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=xy.device)
        xy_u  = xy[iu[0], iu[1]].sqrt()
        Txy_u = Txy[iu[0], iu[1]].sqrt()

        return loss, xy_u.detach().cpu().numpy(), Txy_u.detach().cpu().numpy()

    return loss

def get_n_by_n_images(imgs, nrow=5, ncol=5):
    # check if imgs values within [0,1]
    if torch.any((imgs > 1) | (imgs < 0)):
        imgs = (imgs+1.0)/2.0
    img_size = imgs.shape[2]
    tmp_mat = np.zeros((3,img_size*nrow,img_size*ncol))
    for i in range(nrow):
        for j in range(ncol):
            k = i*ncol + j
            tmp_mat[:,img_size*i:img_size*(i+1),img_size*j:img_size*(j+1)] = imgs[k].detach().cpu()
    tmp_mat = swap_axes_images(tmp_mat)
    return tmp_mat

def create_xy_plot(xdata, T, zdim, dataloader, save_fig_path):
    device = xdata.device
    with torch.no_grad():
        count = 0
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        for _, (imgs, _) in enumerate(tqdm.tqdm(dataloader, leave=False)): 
            x = imgs.to(device)
            _, xx, Txx = compute_GME_cost(T,x,more=True)
            Tx = T(x).detach().cpu()
            ax[0].scatter(Tx[:,0],Tx[:,1])
            ax[1].plot(xx, Txx/xx,'.',alpha=0.2)
            ax[1].set_title(f'sd: {np.std(Txx/xx)}')
            count += 1
            if count > 10:
                break
        plt.savefig(f'{save_fig_path}/00000Txstatus.png')
    
        batch_size = 100
        x = xdata[:batch_size]
        Tx = T(x)
        z = torch.randn(Tx.shape, device=device)
        xy  = ((x.view((x.shape[0],1,-1))-x.view((1,x.shape[0],-1)))**2).sum(2) ** 0.5
        zw  = ((z.view((z.shape[0],1,-1))-z.view((1,z.shape[0],-1)))**2).sum(2) ** 0.5
        Txy = ((Tx.view((Tx.shape[0],1,-1))-Tx.view((1,Tx.shape[0],-1)))**2).sum(2) ** 0.5
        xy  = xy.detach().cpu().numpy()
        zw  = zw.detach().cpu().numpy()
        Txy = Txy.detach().cpu().numpy()
        
        fig,ax = plt.subplots(1,3,figsize=(10,4))
        
        zz = 1.0/(1.0 + zw).mean(1)
        TT = 1.0/(1.0 +Txy).mean(1)
        ax[2].plot(np.sort(zz),label='z')
        ax[2].plot(np.sort(TT),label='Tx')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_title(f'T: {TT.min():9.2e}, {TT.max():9.2e}, {TT.mean():9.2e}\nz: {zz.min():9.2e}, {zz.max():9.2e}, {zz.mean():9.2e}')
        
        xy  = xy[np.triu_indices(x.shape[0],1)]
        zw  = zw[np.triu_indices(x.shape[0],1)]
        Txy = Txy[np.triu_indices(x.shape[0],1)]
        
        ax[0].plot(xy,Txy, '.', alpha=0.1)
        ax[0].set_title('xx and Txy')
        ax[0].grid()
        ax[1].plot(Txy,zw, '.', alpha=0.1)
        ax[1].set_title(f'Tx: {np.min(Txy):.3e}, {np.max(Txy):.3e}, {np.mean(Txy):.3e}\nz : {np.min(zw):.3f}, {np.max(zw):.3f}, {np.mean(zw):.3f}')
        ax[1].grid()
        filename = f'{save_fig_path}/00xyplot.png'
        plt.savefig(filename)
        print(f'xy plot saved in {filename}')
        plt.close('all')

class Parameters:
    def __init__(self, batch_size=100, sample_size=1000, plot_freq=200, MAX_OUTER_ITER=500, zdim=100, scale=1.0):
        self.batch_size  = batch_size 
        self.sample_size = sample_size
        self.plot_freq   = plot_freq
        self.MAX_OUTER_ITER = MAX_OUTER_ITER
        self.zdim = zdim
        self.scale = scale
    def print(self):
        print(f"zdim: {self.zdim}, batch_size: {self.batch_size}")
      
class Pbar:
    def __init__(self, dataloader, use_tqdm=0, desc="", leave=False):
        self.use_tqdm = use_tqdm
        if use_tqdm == 0:    
            self.range_dataloader = dataloader
        else:
            self.range_dataloader = tqdm.tqdm(dataloader, desc=desc, leave=leave)
    def range(self):
        return self.range_dataloader
    def set_description(self, str0):
        if self.use_tqdm == 1:
            self.range_dataloader.set_description(str0)
    def write(self, str0): # only for tqdm
        if self.use_tqdm == 0:
            print(str0)
        else:
            self.range_dataloader.write(str0)
    def __call__(self):
        return self.range()


@torch.no_grad()
def extract_features(data: torch.Tensor, size=(32,32), out_dtype=torch.float32):
    """
    Vectorized resize+flatten-free 'feature' extractor.
    Accepts [N,C,H,W] or [C,H,W]; uint8 [0,255] or float in [0,1] or [-1,1].
    Returns a tensor of shape [N, C, size[0], size[1]] on the same device.
    """
    x = data
    if x.dim() == 3:                     # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)

    # Convert to float in [0,1]
    if x.dtype == torch.uint8:
        x = x.float().div_(255)
    elif x.dtype.is_floating_point:
        # Map [-1,1] -> [0,1] if needed
        if x.min() < -1e-4 or x.max() > 1 + 1e-4:
            x = x.clamp(-1, 1)
            x = (x + 1.0) / 2.0
    else:
        x = x.float()

    # Batch resize on the same device (GPU-accelerated if x is on CUDA)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    # Keep dtype consistent for downstream math
    if x.dtype != out_dtype:
        x = x.to(out_dtype)

    return x.contiguous()



@torch.no_grad()
def pushforward_scaled_chunked(R, z_all, Nt, scale, chunk_size=256, device=None):
    device = device or (z_all.device if z_all.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    R.eval()
    N = z_all.shape[0]

    out = None
    i0 = 0
    while i0 < N:
        i1 = min(i0 + chunk_size, N)
        z_chunk = z_all[i0:i1].to(device)
        Rz = pushforward(R, z_chunk, Nt=Nt)

        if out is None:
            out = torch.empty((N, *Rz.shape[1:]), device=device, dtype=Rz.dtype)

        out[i0:i1].copy_(Rz)               # write into preallocated buffer
        i0 = i1

    return out.detach()

# ===============================
# Streaming FID with your formula + SRz preview
# ===============================
def compute_fid_pixel_streaming_fixed_z(
    R, z_val, real_loader,
    N_real: int,
    Nt_push: int, 
    gen_chunk: int = 256,
    return_preview: int = 25,
    device=None,
):
    """
    Uses a fixed noise tensor z_val (in pixel dimensions) to compute:
      fid = ||mu_real - mu_fake||^2 + tr(S_real + S_fake - 2 * sqrt(S_real  S_fake))
    Returns (fid_value, Rz_vis) where Rz_vis is a small preview batch for plotting.
    
    Assumes you have:
      - extract_features(tensor)
      - calculate_fid(real_features, fake_features)
    """
    R.eval()

    # 1) Collect exactly N_real real images
    imgs_list, n_collected = [], 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        take = min(imgs.size(0), N_real - n_collected)
        if take > 0:
            imgs_list.append(imgs[:take].contiguous())
            n_collected += take
        if n_collected >= N_real:
            break
    if n_collected < N_real:
        raise ValueError(f"real_loader yielded only {n_collected} images, need {N_real}")

    x = torch.cat(imgs_list, dim=0).to(device, non_blocking=True)  
    real_features = extract_features(x)

    # 2) Generate using fixed z_val in chunks, embed per chunk
    N_gen = z_val.size(0)
    fake_feats_list = []
    Rz_vis = None

    for i0 in range(0, N_gen, gen_chunk):
        i1 = min(i0 + gen_chunk, N_gen)
        z_chunk = z_val[i0:i1].to(device, non_blocking=True)

        # In pixel space, pushforward directly outputs the image
        Rz  = pushforward(R, z_chunk, Nt=Nt_push)

        if Rz_vis is None:
            Rz_vis = Rz[:min(return_preview, Rz.size(0))].detach().cpu()

        fake_feats_list.append(extract_features(Rz))

    fake_features = torch.cat(fake_feats_list, dim=0)

    # 3) Compute FID
    fid = calculate_fid(real_features, fake_features)

    return float(fid), Rz_vis

@torch.no_grad()
def compute_fid_hadamard_streaming_fixed_z(
    R, S_net, z_val, real_loader,
    N_real: int,
    Nt_push: int, 
    gen_chunk: int = 256,
    return_preview: int = 25,
    device=None,
):
    """
    Uses a fixed latent tensor z_val to compute:
      fid = ||mu_real - mu_fake||^2 + tr(S_real + S_fake - 2 * sqrt(S_real ⊙ S_fake))
    Returns (fid_value, SRz_vis) where SRz_vis is a small preview batch for plotting.

    Assumes you have:
      - extract_features(tensor, transform_resize)
      - calculate_fid(real_features, fake_features)
    """
    R.eval()

    # 1) Collect exactly N_real real images
    imgs_list, n_collected = [], 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        take = min(imgs.size(0), N_real - n_collected)
        if take > 0:
            imgs_list.append(imgs[:take].contiguous())
            n_collected += take
        if n_collected >= N_real:
            break
    if n_collected < N_real:
        raise ValueError(f"real_loader yielded only {n_collected} images, need {N_real}")

    x = torch.cat(imgs_list, dim=0).to(device, non_blocking=True)  # [N_real, C, H, W]
    real_features = extract_features(x)

    # 2) Generate using fixed z_val in chunks, embed per chunk
    N_gen = z_val.size(0)
    fake_feats_list = []
    SRz_vis = None

    for i0 in range(0, N_gen, gen_chunk):
        i1 = min(i0 + gen_chunk, N_gen)
        z_chunk = z_val[i0:i1].to(device, non_blocking=True)

        Rz  = pushforward(R, z_chunk, Nt=Nt_push)
        SRz = S_net(Rz)

        if SRz_vis is None:
            SRz_vis = SRz[:min(return_preview, SRz.size(0))].detach().cpu()

        fake_feats_list.append(extract_features(SRz))

    fake_features = torch.cat(fake_feats_list, dim=0)

    # 3) Your FID computation
    fid = calculate_fid(real_features, fake_features)

    return float(fid), SRz_vis

@torch.no_grad()
def get_STz_samples(
    R, S_net, z_val,
    Nt_push: int, return_preview: int = 25,
    device=None,
):
    """
    Generating images S(R(z)) for a fixed z_val.
    """
    # Generate images
    z_chunk = z_val[:return_preview].to(device, non_blocking=True)
    Rz  = pushforward(R, z_chunk, Nt=Nt_push)
    SRz = S_net(Rz)
    SRz_vis = SRz[:min(return_preview, SRz.size(0))].detach().cpu()

    return SRz_vis


@torch.no_grad()
def compute_fid_just_vae(
    S_net, z_val, real_loader,
    N_real: int,
    gen_chunk: int = 256,
    return_preview: int = 25,
    device=None,
):
    """
    Uses a fixed latent tensor z_val to compute:
      fid = ||mu_real - mu_fake||^2 + tr(S_real + S_fake - 2 * sqrt(S_real ⊙ S_fake))
    Returns (fid_value, SRz_vis) where SRz_vis is a small preview batch for plotting.

    Assumes you have:
      - extract_features(tensor, transform_resize)
      - calculate_fid(real_features, fake_features)
    """
    # 1) Collect exactly N_real real images
    imgs_list, n_collected = [], 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        take = min(imgs.size(0), N_real - n_collected)
        if take > 0:
            imgs_list.append(imgs[:take].contiguous())
            n_collected += take
        if n_collected >= N_real:
            break
    if n_collected < N_real:
        raise ValueError(f"real_loader yielded only {n_collected} images, need {N_real}")

    x = torch.cat(imgs_list, dim=0).to(device, non_blocking=True)  # [N_real, C, H, W]
    real_features = extract_features(x)

    # 2) Generate using fixed z_val in chunks, embed per chunk
    N_gen = z_val.size(0)
    fake_feats_list = []
    Sz_vis = None

    for i0 in range(0, N_gen, gen_chunk):
        i1 = min(i0 + gen_chunk, N_gen)
        z_chunk = z_val[i0:i1].to(device, non_blocking=True)
        Sz = S_net(z_chunk)
        if Sz_vis is None:
            Sz_vis = Sz[:min(return_preview, Sz.size(0))].detach().cpu()

        fake_feats_list.append(extract_features(Sz))

    fake_features = torch.cat(fake_feats_list, dim=0)

    # 3) Your FID computation
    fid = calculate_fid(real_features, fake_features)

    return float(fid), Sz_vis



# ------------ helper for display range ------------
def to_disp(t: torch.Tensor) -> torch.Tensor:
    # map [-1,1] -> [0,1] and clamp
    return ((t.clamp(-1, 1) + 1.0) / 2.0).contiguous()



def compute_reconstruction_loss_chunked(x_val, T, S, chunk_size=100):
    """
    Computes mean( (x - S(T(x)))^2 ) over x in x_val without O(|x_val|) memory.

    Parameters
    ----------
    x_val : torch.Tensor
        Validation tensor of shape (N, ...). Can be on CPU or GPU.
    T, S : torch.nn.Module
        Models. They will be run in eval mode in this function.
    chunk_size : int
        Number of samples per chunk.

    Returns
    -------
    float
        Scalar validation loss as a Python float.
    """
    # Remember original training modes to restore later
    S_was_training = S.training
    S.eval()

    n = x_val.shape[0]
    total_sq_error = 0.0
    total_count = 0

    # Choose device from the modules
    try:
        device = next(S.parameters()).device
    except StopIteration:
        device = x_val.device  # fallback

    # Inference mode disables autograd and saves memory compared to no_grad
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_chunk = x_val[start:end]

            # Move only the current chunk to the model device
            if x_chunk.device != device:
                x_chunk = x_chunk.to(device, non_blocking=True)

            # Forward
            Tx = T(x_chunk)
            STx = S(Tx)

            # Accumulate sum of squared errors and element count
            # This reproduces .pow(2).mean() over the entire x_val
            se_sum = torch.sum((x_chunk - STx) ** 2)
            total_sq_error += se_sum.item()
            total_count += x_chunk.numel()

            # Free CUDA memory for this chunk
            del x_chunk, Tx, STx
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Restore original modes
    if S_was_training: S.train()

    return total_sq_error / total_count


def build_dataset_and_params(data_str: str, augmentation: bool = True):
    """
    Returns:
      dataset_nn, PARAM, img_size, xshape

    New:
      augmentation (bool):
        If True, apply mild augmentations at data loading time:
          - random horizontal flip
          - slight random resized crop (mild zoom/crop)
          - random gaussian blur

    Notes:
      - Augmentations are applied in the dataset's transform pipeline (PIL-space),
        before ToTensor/Normalize.
      - For MNIST, flips are usually not label-preserving. Since you asked for the
        listed augmentations, this still enables them if augmentation=True,
        but you may want to keep augmentation=False for MNIST in practice.
    """
    data_str = data_str.lower()

    def make_transform(
        *,
        img_size: int,
        is_gray: bool,
        do_augment: bool,
    ) -> transforms.Compose:
        # Base preprocessing (resize/crop) + optional augmentation + tensor + normalize
        tfs = []

        if do_augment:
            # "slight resize": mild RandomResizedCrop (close to original size)
            tfs.append(
                transforms.RandomResizedCrop(
                    size=img_size,
                    scale=(0.95, 1.0),
                    ratio=(0.95, 1.05),
                    antialias=True,
                )
            )
            # side-way flip
            tfs.append(transforms.RandomHorizontalFlip(p=0.5))

            # gaussian blur (mild, applied with small probability)
            # kernel must be odd
            # k = 3 if img_size >= 64 else 3
            # tfs.append(
            #     transforms.RandomApply(
            #         [transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))],
            #         p=0.15,
            #     )
            # )
        else:
            # Deterministic preprocessing
            tfs.append(transforms.Resize(img_size))
            tfs.append(transforms.CenterCrop(img_size))

        tfs.append(transforms.ToTensor())

        if is_gray:
            tfs.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            tfs.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        return transforms.Compose(tfs)

    if data_str == "mnist":
        from transportmodules.transportsMNIST import TransportG, TransportT  # noqa: F401

        data_dir = "../data/mnist"
        img_size = 32
        xshape = (1, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=True, do_augment=augmentation)

        dataset_nn = datasets.MNIST(data_dir, download=True, transform=transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=30, scale=1.0)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "cifar":
        from transportmodules.transportsCifar import TransportG, TransportT  # noqa: F401

        data_dir = "../data/cifar"
        img_size = 32
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = datasets.CIFAR10(data_dir, download=True, transform=transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=50, scale=1.0)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celeb":
        from transportmodules.transportsCeleb import TransportG, TransportT  # noqa: F401

        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = CelebADataset(data_dir, transform)
        # dataset_nn = datasets.CelebA(root='../data', split='train', transform=transform, download=True)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celeb2": # lightweight version
        from transportmodules.transportsCeleb2 import TransportG, TransportT  # noqa: F401

        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = CelebADataset(data_dir, transform)
        # dataset_nn = datasets.CelebA(root='../data', split='train', transform=transform, download=True)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celeb3": # lightweight version
        from transportmodules.transportsCeleb3 import TransportG, TransportT  # noqa: F401

        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = CelebADataset(data_dir, transform)
        # dataset_nn = datasets.CelebA(root='../data', split='train', transform=transform, download=True)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celeb4": # lightweight version
        from transportmodules.transportsCeleb4 import TransportG, TransportT  # noqa: F401

        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = CelebADataset(data_dir, transform)
        # dataset_nn = datasets.CelebA(root='../data', split='train', transform=transform, download=True)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=512, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape


    if data_str == "ffhq":
        # NOTE: you currently point FFHQ to celebA. Keep as-is unless you have a separate FFHQ path.
        data_dir = (Path.cwd().parent / "data" / "ffhqimages" / "thumbnails128x128" ).resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = FFHQDataset(data_dir, transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=512, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "ffhq2":
        # NOTE: you currently point FFHQ to celebA. Keep as-is unless you have a separate FFHQ path.
        data_dir = (Path.cwd().parent / "data" / "ffhqimages" / "thumbnails128x128" ).resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = FFHQDataset(data_dir, transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celebahq": # heavy architecture
        from transportmodules.transportsCelebHQ import TransportG, TransportT  # noqa: F401

        data_dir = (Path.cwd().parent / "data" / "celeba_hq_256" / "images").resolve()
        img_size = 256
        xshape = (3, img_size, img_size)

        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)

        dataset_nn = CustomCelebAHQ(data_dir, transform=transform)
        PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=100, zdim=100, scale=0.1)
        return dataset_nn, PARAM, img_size, xshape


    # -------------------------
    # NEW: CelebA-HQ resized to 64x64
    # -------------------------
    if data_str == "celebahq64":
        from transportmodules.transportsCeleb import TransportG, TransportT  # noqa: F401
        data_dir = (Path.cwd().parent / "data" / "celeba_hq_256" / "images").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)
        transform = make_transform(img_size=img_size, is_gray=False, do_augment=augmentation)
        dataset_nn = CustomCelebAHQ(data_dir, transform=transform)

        # You can keep these, but typical to increase batch vs 256 case.
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=1000, scale=0.3) # the original zdim=100 but I changed it (4/3/2026) to address reviewer's comment
        return dataset_nn, PARAM, img_size, xshape


    print(f"Unknown dataset: {data_str}")
    sys.exit(1)

def get_transport_classes(data_str: str):
    """
    Import TransportT / TransportG from the correct module depending on dataset.
    This matches your original imports.
    """
    data_str = data_str.lower()
    if data_str == "mnist":
        from transportmodules.transportsMNIST import TransportG, TransportT, NetSingle
    elif data_str == "cifar":
        from transportmodules.transportsCifar import TransportG, TransportT, NetSingle
    elif data_str == "celeb":
        from transportmodules.transportsCeleb import TransportG, TransportT, NetSingle
    elif data_str == "celeb2":
        from transportmodules.transportsCeleb2 import TransportG, TransportT, NetSingle
    elif data_str == "celeb3":
        from transportmodules.transportsCeleb3 import TransportG, TransportT, NetSingle
    elif data_str == "celeb4":
        from transportmodules.transportsCeleb4 import TransportG, TransportT, NetSingle
    elif data_str == "ffhq":
        from transportmodules.transportsFFHQ import TransportG, TransportT, NetSingle
    elif data_str == "ffhq2":
        from transportmodules.transportsFFHQ import TransportG, TransportT, NetSingle
    elif data_str == "celebahq":
        from transportmodules.transportsCelebHQ import TransportG, TransportT, NetSingle
    elif data_str == "celebahq64":
        from transportmodules.transportsCeleb import TransportG, TransportT, NetSingle
    else:
        raise ValueError(f"Unknown dataset for transports: {data_str}") #hello

    return TransportT, TransportG, NetSingle


@torch.inference_mode()
def compute_scalar_scale_C_from_dataloader(
    dataloader,
    encoder,
    *,
    max_samples = 5000,
    device = None,
    eps = 1e-12,
) -> Tuple[float, torch.Tensor, dict]:
    """
    Compute C = sqrt(D / E||T(x) - E[T(x)]||^2) using streaming statistics.

    Efficient version:
      - encoder forward pass happens on `device` (typically GPU)
      - statistics are accumulated on CPU in float64
      
    Returns:
        C (float): The computed scalar scale.
        mean (torch.Tensor): The computed mean of T(x) across the dataset (shape D, float32).
        stats (dict): Additional statistics from the calculation.
    """
    encoder_was_training = encoder.training
    encoder.eval()

    n = 0
    mean_cpu = None            # shape (D,), float64 on CPU
    m2_sum = 0.0               # scalar float, sum of squared deviations

    for batch in dataloader:
        if max_samples is not None and n >= max_samples:
            break

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not torch.is_tensor(x):
            raise TypeError(f"Expected tensor batch or (tensor, ...), got {type(x)}")

        if device is not None:
            x = x.to(device, non_blocking=True)

        Tx = encoder(x)                       # (B, ...)
        Tx = Tx.reshape(Tx.shape[0], -1)      # (B, D)

        # Respect max_samples
        B = Tx.shape[0]
        if max_samples is not None and n + B > max_samples:
            B_keep = max_samples - n
            Tx = Tx[:B_keep]
            B = B_keep

        # Move to CPU float64 for numerically stable accumulation
        Tx_cpu = Tx.detach().to("cpu", dtype=torch.float64)

        # Batch mean and within-batch M2
        batch_mean = Tx_cpu.mean(dim=0)  # (D,)
        batch_m2 = ((Tx_cpu - batch_mean).pow(2).sum(dim=1)).sum().item()

        if n == 0:
            mean_cpu = batch_mean.clone()
            m2_sum = batch_m2
            n = B
        else:
            n_new = n + B
            delta = batch_mean - mean_cpu
            # Merge two groups: (n, mean_cpu, m2_sum) and (B, batch_mean, batch_m2)
            m2_sum = m2_sum + batch_m2 + float(delta.pow(2).sum().item()) * (n * B / n_new)
            mean_cpu = mean_cpu + delta * (B / n_new)
            n = n_new

    if encoder_was_training:
        encoder.train()

    if n == 0:
        raise ValueError("No samples were processed.")

    r2 = m2_sum / n
    D_eff = int(mean_cpu.numel())
    C = math.sqrt(float(D_eff) / max(float(r2), eps))

    # Cast the mean back to standard float32 before returning
    mean_out = mean_cpu.to(dtype=torch.float32).to(device)

    stats = {
        "num_samples_used": int(n),
        "D": D_eff,
        "E_norm2_centered": float(r2),
        "C": float(C),
        "mean": mean_out,
    }
    
    return stats

# ----------------------------
# Repro / device helpers
# ----------------------------
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def get_device(force_cuda_flag: bool = False) -> torch.device:
    """
    If force_cuda_flag=True, require CUDA.
    Otherwise use CUDA if available.
    """
    if force_cuda_flag and not torch.cuda.is_available():
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


def set_visible_gpus(gpu_list: str | None) -> None:
    """
    Set CUDA_VISIBLE_DEVICES if user provided a list like "0,1,2".
    If None, do nothing (use the environment default).
    """
    if gpu_list is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


# ----------------------------
# 8x8 corner interpolation plot
# ----------------------------
@torch.inference_mode()
def save_corner_interpolation_8x8(
    *,
    x_corners: torch.Tensor,   # (4, C, H, W) in [0,1] (or whatever your dataset gives)
    T_scale_fn,                # callable: x -> latent (B,zdim)
    S: torch.nn.Module,        # decoder: latent -> image
    R: torch.nn.Module,        # velocity field
    out_path: Path,
    device: torch.device,
    Nt_inv: int = 50,          # steps for inverse flow
    Nt_fwd: int = 50,          # steps for forward flow
    padding: int = 2,
) -> None:
    """
    Build an 8x8 image grid where corners are the given images x1..x4, and all other
    cells are generated by:
      x_i -> T(x_i) -> inverse-flow via R to z_i
      z-grid via bilinear interpolation of z1..z4
      z(u,v) -> forward-flow via R to x(1) -> decode via S
    """
    assert x_corners.ndim == 4 and x_corners.size(0) == 4, "x_corners must be (4,C,H,W)"

    # Make sure everything is on device
    x_c = x_corners.to(device)

    # Encode corners
    Tx = T_scale_fn(x_c)  # (4,zdim)

    # Inverse flow to get z1..z4
    R_was_training = R.training
    R.eval()

    z_corners = pushforward_inv(R, Tx, Nt=Nt_inv)  # (4,zdim)
    z1, z2, z3, z4 = z_corners[0], z_corners[1], z_corners[2], z_corners[3]

    # Build 8x8 latent grid by bilinear interpolation in z-space.
    # Convention:
    #   (row=0,col=0) is z1  (top-left)
    #   (row=0,col=7) is z2  (top-right)
    #   (row=7,col=0) is z3  (bottom-left)
    #   (row=7,col=7) is z4  (bottom-right)
    n = 6
    us = torch.linspace(0.0, 1.0, n, device=device, dtype=z1.dtype)  # horizontal
    vs = torch.linspace(0.0, 1.0, n, device=device, dtype=z1.dtype)  # vertical

    z_grid_list = []
    for r in range(n):
        v = vs[r]
        for c in range(n):
            u = us[c]
            z_uv = (1 - u) * (1 - v) * z1 + u * (1 - v) * z2 + (1 - u) * v * z3 + u * v * z4
            z_grid_list.append(z_uv)
    z_grid = torch.stack(z_grid_list, dim=0)  # (64, zdim)

    # Push forward through R to get latent at t=1, then decode
    x1_lat = pushforward(R, z_grid, Nt=Nt_fwd)  # (64,zdim)
    imgs_gen = S(x1_lat)  # (64,C,H,W)

    # Convert both generated images and corners to the SAME display space
    imgs_disp = to_disp(imgs_gen)   # (64,C,H,W) in [0,1]
    corners_disp = to_disp(x_c)     # (4,C,H,W) in [0,1]

    # Now overwrite corners (in display space)
    idx_tl = 0
    idx_tr = n - 1
    idx_bl = (n - 1) * n
    idx_br = n * n - 1
    imgs_disp[idx_tl] = corners_disp[0]
    imgs_disp[idx_tr] = corners_disp[1]
    imgs_disp[idx_bl] = corners_disp[2]
    imgs_disp[idx_br] = corners_disp[3]

    grid = make_grid(imgs_disp, nrow=n, padding=padding)
    save_image(grid, out_path)

    # restore modes (optional)
    if R_was_training:
        R.train()



from torchvision.models import inception_v3, Inception_V3_Weights
from pytorch_fid.inception import InceptionV3

# ===============================
# Standard FID Utilities
# ===============================
import os
import urllib.request
import pickle
import torch
import torch.nn as nn

# ===============================
# Standard FID Utilities
# ===============================
import torch
import torch.nn as nn
from pytorch_fid.inception import InceptionV3

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device, input_range="[-1, 1]"):
        super().__init__()
        print('Loading standard Inception-v3 model from pytorch-fid...')
        
        # 2048 is the standard feature dimension for FID
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        
        # This automatically downloads and loads the correct legacy weights!
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
        self.device = device
        
        # Store the expected input range to apply consistent scaling
        valid_ranges = ["[-1, 1]", "[0, 1]", "[0, 255]"]
        if input_range not in valid_ranges:
            raise ValueError(f"input_range must be one of {valid_ranges}")
        self.input_range = input_range

    @torch.no_grad()
    def forward(self, x):
        """
        Expects x to be [B, C, H, W].
        Deterministically maps known input ranges to [0, 1] for Inception.
        """
        
        # 1. Deterministic Normalization to [0, 1]
        if self.input_range == "[0, 255]":
            x = x / 255.0
        elif self.input_range == "[-1, 1]":
            x = (x + 1.0) / 2.0
            
        # Final clamp to guarantee strict bounds (handles minor float imprecision)
        x = x.clamp(0.0, 1.0)

        # 2. Ensure 3 channels (Grayscale to RGB)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 3. Extract features
        # pytorch-fid inherently handles the 299x299 resizing internally
        # and applies the ImageNet mean/std normalization.
        features = self.model(x)[0]
        
        # Flatten spatial dimensions: [B, 2048, 1, 1] -> [B, 2048]
        features = features.squeeze(3).squeeze(2)
        
        return features


def calculate_fid(real_embeddings, generated_embeddings):
    real_embeddings      = real_embeddings.reshape((real_embeddings.shape[0],-1))
    generated_embeddings = generated_embeddings.reshape((generated_embeddings.shape[0],-1))
    mu1, sigma1 = torch.mean(real_embeddings, dim=0), torch.cov(real_embeddings.t())
    mu2, sigma2 = torch.mean(generated_embeddings, dim=0), torch.cov(generated_embeddings.t())
    # Calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    # Calculate square root of the product between covariances
    covmean = torch.sqrt(torch.mul(sigma1, sigma2))
    # Convert to real part if complex
    if covmean.is_complex():
        covmean = covmean.real
    # Calculate FID score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid.item()  # Convert to scalar value



import torch
import numpy as np
import scipy.linalg

def compute_frechet_stats(features: torch.Tensor):
    """
    Computes mu and sigma using PyTorch batched matrix operations.
    Uses mean-centering to completely avoid floating-point cancellation errors.
    """
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError("Features contain NaNs or Infs! Check your pipeline.")
        
    # FLATTEN spatial dimensions if they exist
    features = features.reshape(features.shape[0], -1)
        
    N = features.shape[0]
    if N <= 1:
        raise ValueError(f"Batch size must be > 1 to compute covariance. Got {N}")

    # Force float64 for deep numerical stability
    features = features.to(torch.float64)
    
    # 1. Calculate the mean
    mu = features.mean(dim=0)
    
    # 2. Mean-center the features FIRST
    centered_features = features - mu
    
    # 3. Compute covariance on the centered features
    sigma = (centered_features.T @ centered_features) / (N - 1)

    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    """
    Computes the Fréchet Distance from precalculated mu and sigma.
    Includes the robust SciPy numerical stability checks from the original code.
    """
    m = np.square(mu - mu_ref).sum()
    cov_prod = np.dot(sigma, sigma_ref)
    s, _ = scipy.linalg.sqrtm(cov_prod, disp=False)

    # --- Standard Numerical Stability Checks ---
    if not np.isfinite(s).all():
        print("FID calculation produced singular product; adding 1e-6 to the diagonal.")
        offset = np.eye(sigma.shape[0]) * 1e-6
        s, _ = scipy.linalg.sqrtm(np.dot(sigma + offset, sigma_ref + offset), disp=False)

    if np.iscomplexobj(s):
        if not np.allclose(np.diagonal(s).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(s.imag))
            raise ValueError(f"Imaginary component {max_imag} is too high.")
        s = s.real

    fid = m + np.trace(sigma + sigma_ref - s * 2.0)
    return float(np.real(fid))


# ===============================
# Inception FID: Gen & Recon
# ===============================
@torch.no_grad()
def compute_fid_inception_gen_and_recon(
    R, S_net, T_net, z_val, real_loader,
    N_real: int,
    Nt_push: int, 
    gen_chunk: int = 256,
    return_preview: int = 25,
    device=None,
):
    """
    Computes standard FID (using Inception features) for both:
      1) Generated images: S_net(pushforward(R, z))
      2) Reconstructed images: S_net(T_net(x))
    """
    R.eval()
    S_net.eval()

    # Load inception model dynamically and move to device
    inception = InceptionFeatureExtractor(device)

    # 1) Process Generated Images in Chunks
    N_gen = z_val.size(0)
    gen_feats_list = []
    gen_feats_pixel_list = []
    gen_vis = None

    for i0 in range(0, N_gen, gen_chunk):
        i1 = min(i0 + gen_chunk, N_gen)
        z_chunk = z_val[i0:i1].to(device, non_blocking=True)

        # Generate: decode(pushforward(z))
        Rz  = pushforward(R, z_chunk, Nt=Nt_push)
        SRz = S_net(Rz)

        if gen_vis is None:
            gen_vis = SRz[:min(return_preview, SRz.size(0))].detach().cpu()

        if torch.isnan(Rz).any().item():
            continue

        gen_feats_list.append(inception(SRz))
        gen_feats_pixel_list.append(extract_features(SRz))

    gen_features = torch.cat(gen_feats_list, dim=0)
    gen_pixel_features = torch.cat(gen_feats_pixel_list, dim=0)

    # 2) Collect exactly N_real real images
    imgs_list, n_collected = [], 0
    for batch in real_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        take = min(imgs.size(0), N_real - n_collected)
        if take > 0:
            imgs_list.append(imgs[:take].contiguous())
            n_collected += take
        if n_collected >= N_real:
            break
            
    if n_collected < N_real:
        raise ValueError(f"real_loader yielded only {n_collected} images, need {N_real}")

    x_all = torch.cat(imgs_list, dim=0)

    # 3) Process Real and Reconstructed in Chunks
    real_feats_list = []
    recon_feats_list = []
    real_feats_pixel_list = []
    recon_vis = None

    for i0 in range(0, N_real, gen_chunk):
        i1 = min(i0 + gen_chunk, N_real)
        x_chunk = x_all[i0:i1].to(device, non_blocking=True)

        real_feats_list.append(inception(x_chunk))
        real_feats_pixel_list.append(extract_features(x_chunk))

        # Reconstruct: decode(encode(x))
        z_recon = T_net(x_chunk)
        x_recon = S_net(z_recon)

        if recon_vis is None:
            recon_vis = x_recon[:min(return_preview, x_recon.size(0))].detach().cpu()

        recon_feats_list.append(inception(x_recon))

    real_features = torch.cat(real_feats_list, dim=0)
    real_pixel_features = torch.cat(real_feats_pixel_list, dim=0)
    recon_features = torch.cat(recon_feats_list, dim=0)

    # 4) Compute Stats and FIDs using the new logic
    # Calculate stats for the required feature tensors
    mu_real, sigma_real = compute_frechet_stats(real_features)
    mu_gen, sigma_gen   = compute_frechet_stats(gen_features)
    mu_recon, sigma_recon = compute_frechet_stats(recon_features)
    
    mu_real_pix, sigma_real_pix = compute_frechet_stats(real_pixel_features)
    mu_gen_pix, sigma_gen_pix   = compute_frechet_stats(gen_pixel_features)

    # Calculate final FIDs
    fid_gen   = calculate_fid_from_inception_stats(mu_real, sigma_real, mu_gen, sigma_gen)
    fid_recon = calculate_fid_from_inception_stats(mu_recon, sigma_recon, mu_gen, sigma_gen)
    fid_pixel = calculate_fid_from_inception_stats(mu_real_pix, sigma_real_pix, mu_gen_pix, sigma_gen_pix)

    fid_tmp   = calculate_fid_from_inception_stats(mu_real, sigma_real, mu_recon, sigma_recon)
    print(f"sanity: {fid_tmp}")

    
    # Free up VRAM from Inception model
    del inception
    torch.cuda.empty_cache()

    return float(fid_gen), float(fid_recon), float(fid_pixel), gen_vis
