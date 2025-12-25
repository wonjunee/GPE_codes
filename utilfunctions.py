from typing import Tuple
import argparse
import sys
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
import torch
from PIL import Image
from natsort import natsorted
from pathlib import Path
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import contextlib

from PIL import Image, ImageFilter, ImageOps

class AddBlurredBorder:
    def __init__(self, border_size=4, blur_radius=4):
        self.border_size = border_size
        self.blur_radius = blur_radius

    def __call__(self, img):
        # Create a border around the image
        bordered_image = ImageOps.expand(img, border=self.border_size, fill=0)
        
        # Create a blurred version of the bordered image
        blurred_image = bordered_image.filter(ImageFilter.GaussianBlur(self.blur_radius))
        
        # Paste the original image on top of the blurred image
        bordered_image.paste(img, (self.border_size, self.border_size))
        
        return bordered_image

def get_latent_samples(shape, device):
    return torch.randn(shape, device=device)
    
def swap_axes_images(img):
    img = np.transpose(img, (1,2,0))
    return img

def preprocessing_data(dataset_nn, device, sample_size = 1000, reduced=32):
    """Getting ready with the dataset: x validation set and precomputed mu and sigma for FID.

    Args:
        dataset_nn (torch dataset): Dataset given (MNIST, CIFAR10, CelebA, etc)
        device (str): cpu or gpu
        sample_size (int, optional): the number of samples for validation. Defaults to 1000.

    Returns:
        torch vectors, dict: x validations with a size sample_size, dictionary containing arrays of computed mu and sigma.
    """
    ## Load the dataset 
    dataloader2= torch.utils.data.DataLoader(dataset_nn, batch_size=sample_size, shuffle=True, drop_last=True)
    
    for _, (imgs, _) in enumerate(dataloader2):
        x_val = imgs.to(device)
        img_size = x_val.shape[2]
        break
    
    def calculate_fid_compute_mu_sigma(real_embeddings):
        real_embeddings      = real_embeddings.reshape((real_embeddings.shape[0],-1))
        mu, sigma = torch.mean(real_embeddings, dim=0), torch.cov(real_embeddings.t())
        return mu, sigma

    # compute mu and sigma of the real data in the beginning
    real_mu_sigma = {'mu':[], 'sigma':[]}
    img_skip = img_size // reduced
    for i, (imgs, _) in enumerate(dataloader2):
        x = imgs.to(device) # Images
        mu, sigma = calculate_fid_compute_mu_sigma(x[:,:,::img_skip,::img_skip])
        real_mu_sigma['mu'].append(mu)
        real_mu_sigma['sigma'].append(sigma)
        if i > 10:
            break
    del dataloader2
    return x_val, real_mu_sigma

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

def calculate_fid2(real_mu_sigma, i, generated_embeddings):
    generated_embeddings = generated_embeddings.reshape((generated_embeddings.shape[0],-1))
    mu1, sigma1 = real_mu_sigma['mu'][i], real_mu_sigma['sigma'][i]
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
        
  

class CelebADataset2(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        # image_names0 = os.listdir(root_dir)
        # image_names = glob.glob(f'{root_dir}/*.jpg')
        image_names = []
        for it in os.listdir(root_dir):
            if '.jpg' in it:
                image_names.append(it)
        self.root_dir = root_dir
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img, 0

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


class Velocity(torch.nn.Module):
    def __init__(self):
        super(Velocity, self).__init__()

    def get_close_samples(self,z,x):
        t = torch.rand(x.shape[0], 1, device=x.device)
        z_noise = z + torch.randn(x.shape, device=x.device) * 0.3
        x_z = (z_noise.reshape((z.shape[0],1,z.shape[1])) - x.reshape((1,x.shape[0],x.shape[1]))).pow(2).mean(2)
        ind = torch.argmin(x_z, dim=1)
        x_ind = x[ind]
        xt = (1-t)*z + t*x_ind
        ut = x_ind - z
        return t.detach(), xt.detach(), ut.detach()
    
    def compute_u1(self, x, z, sigma, t=0):
        return (x-z) * (1.0 + t * (1.0/sigma - 1)) + (1. - t) * z, (x-z) * (1.0/sigma - 1) - z
    
    def compute_u1_xz(self, x, z, t):
        return (1-t)*z + t*x, x - z
    

    def compute_u1_xz2(self, x1, x2, z1, z2, t):
        d11 = (x1-z1).pow(2).mean(1)
        d12 = (x1-z2).pow(2).mean(1)
        d21 = (x2-z1).pow(2).mean(1)
        d22 = (x2-z2).pow(2).mean(1)
        
        mask = (d11 + d22 < d12 + d21) * 1
        mask = mask.reshape((-1,1))
        # 1-1, 2-2
        xz1, u1 = ((1-t)*z1 + t*x1)*mask + ((1-t)*z1 + t*x2)*(1-mask), (x1 - z1)*mask +  (x2 - z1)*(1-mask)
        xz2, u2 = ((1-t)*z2 + t*x2)*mask + ((1-t)*z2 + t*x1)*(1-mask), (x2 - z2)*mask +  (x1 - z2)*(1-mask)
        return xz1, xz2, u1, u2 
    
    def compute_velocity_cost_new(self, net, x, z, C=0):
        t = torch.rand(x.shape[0],1,device=x.device)
        x1,u1 = self.compute_u1_xz(x.detach(), z.detach(), t)
        xx = x1.detach() + C * (1 - (2*(t-0.5)).pow(2)) * (torch.rand(x.shape,device=x.device)-0.5)
        value = (net(xx.detach(), t) - u1.detach()).pow(2).mean()
        return value

    def compute_velocity_cost_bfm(self, net, x, z):
        xx = x.detach()
        zz = z.detach()
        value = 0
        t = torch.rand(x.shape[0],1,device=x.device)
        xz, u = self.compute_u1_bfm(x, z, t)
        value += (net(xz.detach(),t) - u.detach()).pow(2).mean()
        return value
    

    def compute_velocity_cost_new_ot(self, net, x, z, C=0.0):
        x1 = x[:x.shape[0]//2,:].detach()
        x2 = x[x.shape[0]//2:,:].detach()
        z1 = z[:z.shape[0]//2,:].detach()
        z2 = z[z.shape[0]//2:,:].detach()
        t = torch.rand(x1.shape[0],1,device=x1.device)
        eps = C * (1 - (2*(t-0.5)).pow(2)) * (torch.rand(x1.shape,device=x1.device)-0.5)
        eps = eps.detach()
        xz1,xz2,u1,u2 = self.compute_u1_xz2(x1, x2, z1, z2, t)
        value = (net(xz1.detach()+eps,t) - u1.detach()).pow(2).mean() + (net(xz2.detach()+eps,t) - u2.detach()).pow(2).mean()
        return value
    
    def forward(self, net, x, inv=False, Nt=None):
        xx = x.detach()
        tau = 1./Nt
        
        if inv:
            x1 = xx
            for i in range(Nt):
                x1 = odeint(net, x1, 1-i * tau, 1-(i+1) * tau)
            xresult = x1
        else:
            x0 = xx
            for i in range(Nt):
                x0 = odeint(net, x0, i * tau, (i+1) * tau)
            xresult = x0
        return xresult
    
    def interpolate(self, net, x, Nt=5, inv=False):
        xx = x * 1.0
        tau = 1.0/Nt
        arr = torch.zeros((Nt+1,x.shape[0],x.shape[1]))
        
        if inv==True:
            arr[-1] = xx.detach()
            x1 = xx
            for i in range(Nt):
                v1 = net(x1,1-i/Nt)
                x1 = x1 - v1 * tau # + torch.randn(x1.shape, device=x1.device) * 0.1
                arr[-1-i-1] = x1.detach()
        else:
            arr[0] = xx.detach()
            x1 = xx
            for i in range(Nt):
                v1 = net(x1,i/Nt)
                x1 = x1 + v1 * tau # + torch.randn(x1.shape, device=x1.device) * 0.1
                arr[i+1] = x1.detach()
        return arr    
    
        
def pushforward(net, x0, Nt=10):
    tau = 1./Nt
    for i in range(Nt):
        x0 = odeint(net, x0.detach(), i * tau, (i+1) * tau)
    return x0.detach()

def pushforward_inv(net, x0, Nt=10):
    tau = 1./Nt
    for i in range(Nt):
        x0 = odeint(net, x0.detach(), 1 - i * tau, 1 - (i+1) * tau)
    return x0.detach()
    
def odeint(odefun, z, t0, t1):
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
    
class Net2(torch.nn.Module):
    def __init__(self, input_dim = 100, output_dim = 100):
        super(Net2, self).__init__()
        dd = 1000
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, output_dim),
        )

    def forward(self, x):
        return self.model(x)
    
class Net_time(torch.nn.Module):
    def __init__(self, input_dim = 100, output_dim = 100):
        super(Net_time, self).__init__()
        dd = 1000
        
        ss = 4
        scale = 10*10*10
        dd = 500
        self.model = nn.Sequential(
                nn.Linear(input_dim+1, ss*scale),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Unflatten(1, (1, ss*scale)),
                nn.Conv1d(1, dd, kernel_size=11, stride=10, padding=5),
                nn.LeakyReLU(0.2, inplace=True), # 1000
                nn.Conv1d(dd, dd*2, kernel_size=11, stride=10, padding=5),
                nn.LeakyReLU(0.2, inplace=True), # 100
                nn.Conv1d(dd*2, dd*4, kernel_size=11, stride=10, padding=5),
                nn.LeakyReLU(0.2, inplace=True), # 10
                nn.Conv1d(dd*4, 1, kernel_size=ss, stride=1, padding=0),
                nn.Flatten(),
        )

    def forward(self, x, t):
        if isinstance(t, float):
            t = torch.ones(x.shape[0],1, device=x.device) * t
        return self.model(torch.cat((x,t),1))

class Net(torch.nn.Module):
    def __init__(self, input_dim = 100, output_dim = 100):
        super(Net, self).__init__()
        dd = 500
        
        self.model = nn.Sequential(
            nn.Linear(input_dim+1, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, dd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dd, output_dim),
        )

    def forward(self, x, t):
        if isinstance(t, float):
            t = torch.ones(x.shape[0],1,device=x.device)
        xt = torch.cat((x,t),1)
        return self.model(xt)

import torch
import torch.nn.functional as F

@torch.no_grad()
def extract_features(data: torch.Tensor, size=(32, 32), out_dtype=torch.float32):
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
    inv_scale = 1.0 / float(scale)

    out = None
    i0 = 0
    while i0 < N:
        i1 = min(i0 + chunk_size, N)
        z_chunk = z_all[i0:i1].to(device)
        Rz = pushforward(R, z_chunk, Nt=Nt)
        Rz.mul_(inv_scale)                 # in-place scale, avoids new allocation

        if out is None:
            out = torch.empty((N, *Rz.shape[1:]), device=device, dtype=Rz.dtype)

        out[i0:i1].copy_(Rz)               # write into preallocated buffer
        i0 = i1

    return out.detach()

# ===============================
# Streaming FID with your formula + SRz preview
# ===============================
import torch
import torch

@torch.no_grad()
def compute_fid_hadamard_streaming_fixed_z(
    R, S_net, z_val, real_loader,
    N_real: int,
    Nt_push: int, scale: float,
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
    inv_scale = 1.0 / float(scale)

    for i0 in range(0, N_gen, gen_chunk):
        i1 = min(i0 + gen_chunk, N_gen)
        z_chunk = z_val[i0:i1].to(device, non_blocking=True)

        Rz  = pushforward(R, z_chunk, Nt=Nt_push)
        Rz.mul_(inv_scale)
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


def build_dataset_and_params(data_str: str):
    """
    Returns:
      dataset_nn, PARAM, img_size, xshape
    """
    data_str = data_str.lower()

    if data_str == "mnist":
        from transportmodules.transportsMNIST import TransportG, TransportT  # noqa: F401

        data_dir = "../data/mnist"
        img_size = 32
        xshape = (1, img_size, img_size)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        dataset_nn = datasets.MNIST(data_dir, download=True, transform=transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=30, scale=1.0)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "cifar":
        from transportmodules.transportsCifar import TransportG, TransportT  # noqa: F401

        data_dir = "../data/cifar"
        img_size = 32
        xshape = (3, img_size, img_size)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset_nn = datasets.CIFAR10(data_dir, download=True, transform=transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=50, scale=1.0)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celeb":
        from transportmodules.transportsCeleb import TransportG, TransportT

        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset_nn = CelebADataset(data_dir, transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "ffhq":
        from transportmodules.transportsCeleb import TransportG, TransportT

        # NOTE: original code uses celebA train path for both celeb and ffhq.
        # If FFHQ is separate in your setup, change this path.
        data_dir = (Path.cwd().parent / "data" / "celebA" / "train").resolve()
        img_size = 64
        xshape = (3, img_size, img_size)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset_nn = CelebADataset(data_dir, transform)
        PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100, zdim=100, scale=0.3)
        return dataset_nn, PARAM, img_size, xshape

    if data_str == "celebahq":
        from transportmodules.transportsCelebHQ import TransportG, TransportT

        data_dir = (Path.cwd().parent / "data" / "celeba_hq_256").resolve()
        img_size = 256
        xshape = (3, img_size, img_size)

        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset_nn = CustomCelebAHQ(data_dir, transform=transform)
        PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=100, zdim=100, scale=0.1)
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
    if data_str == "cifar":
        from transportmodules.transportsCifar import TransportG, TransportT, NetSingle
    if data_str == "celeb":
        from transportmodules.transportsCeleb import TransportG, TransportT, NetSingle
    if data_str == "ffhq":
        from transportmodules.transportsCeleb import TransportG, TransportT, NetSingle
    if data_str == "celebahq":
        from transportmodules.transportsCelebHQ import TransportG, TransportT, NetSingle

    return TransportT, TransportG, NetSingle

    raise ValueError(f"Unknown dataset for transports: {data_str}")



@torch.inference_mode()
def compute_scalar_scale_C_from_dataloader(
    dataloader,
    encoder,
    *,
    max_samples = 5000,
    device = None,
    eps = 1e-12,
) -> Tuple[float, dict]:
    """
    Compute C = sqrt(D / E||T(x) - E[T(x)]||^2) using streaming statistics.

    Efficient version:
      - encoder forward pass happens on `device` (typically GPU)
      - statistics are accumulated on CPU in float64
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

    stats = {
        "num_samples_used": int(n),
        "D": D_eff,
        "E_norm2_centered": float(r2),
        "C": float(C),
    }
    return float(C), stats



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
