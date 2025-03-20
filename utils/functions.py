import argparse
import sys
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import torch
from PIL import Image
from natsort import natsorted
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

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
            # point[0] = x * scale * (0.6 + np.random.normal()*0.3)
            # point[1] = y * scale * (0.6 + np.random.normal()*0.3)

            point[0] = x * 1.5
            point[1] = y * 1.5
            self.centers.append(point)
            
        n_mode2 = 2*n_mode
        for i in range(n_mode2):
            x = np.cos(2.0*np.pi*(i+0.5)/n_mode2)
            y = np.sin(2.0*np.pi*(i+0.5)/n_mode2)
            point = np.random.randn(dim) * 0.2
            # point[0] = x * scale * (0.6 + np.random.normal()*0.3)
            # point[1] = y * scale * (0.6 + np.random.normal()*0.3)

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
    
    
class Velocity2(torch.nn.Module):
    def __init__(self, net, Nt = 10):
        super(Velocity2, self).__init__()
        self.zdim = net.zdim
        self.Nt = Nt
        self.net = net
        
    def compute_u1(self, x, z, sigma, t=0):
        # return (x-z) * (1.0 + t * (1.0/sigma - 1)) + (1. - t) * z, (x-z) * (1.0/sigma - 1) - z
        
        m = 1.0
        k = 1/np.log(1+m) * (1.0/sigma - 1)
        return (x - z) * (1.0 + k * np.log(1+m*t)) + (1. - t) * z, (x-z) * k*m/(1 + m*t) - z
    
        # k = 1.0
        # return (x-z) * (1. - k * t * np.log(t) + t*(1./sigma-1.)) + (1.0 - t) * z, (x-z) * (- k* (np.log(t) + 1) + (1./sigma-1.)) - z
    
    def compute_velocity_cost(self, x, size=1, t=None):
        sigma_min = 1e-3
        xx = x.detach()
        value = 0
        if t == None:
            random_indices = np.random.choice(np.arange(0,self.Nt+1), size=size, replace=False)
        else:
            random_indices = [self.Nt]
        for i in random_indices:
            xx_noise = xx + get_latent_samples(xx.shape,xx.device) * sigma_min
            xx_noise = xx_noise.detach()
            t = i/self.Nt
            x1, u1 = self.compute_u1(xx_noise, xx, sigma_min, t=t)
            
            noise = torch.randn(u1.shape, device=u1.device).detach()
            value += (self.net(x1.detach(),t=1.0-t) + u1.detach() + noise * 0.2 * (0.5 - np.abs(t - 0.5))).pow(2).mean()
        return value
    
    
    def compute_velocity_R_cost(self, z, size=1):
        value = 0
        random_indices = np.random.choice(np.arange(1,self.Nt+1), size=size, replace=False)
        v0 = self.net(z.detach(), t=0.0)
        for i in random_indices:
            t = 1.0 * i/self.Nt
            value += (self.net((z + v0 * t).detach(),t=t) - v0).pow(2).mean()
        return value
    
    def compute_velocity_R_cost2(self, z, R2, size=1):
        value = 0
        random_indices = np.random.choice(np.arange(1,self.Nt+1), size=size, replace=False)
        v0 = R2.net(z.detach(), t=0.0).detach()
        for i in random_indices:
            t = 1.0 * i/self.Nt
            value += (self.net((z + v0 * t).detach(),t=t) - v0).pow(2).mean()
        return value
     
    def forward(self, x, running=False, inv=False, Nt=None):
        
        if Nt == None:
            Nt = self.Nt
            
        xx = x.detach()
        value = torch.zeros((1),device=x.device).mean()
        tau = 1./Nt
        
        if inv:
            x1 = xx
            if running:
                v1 = self.net(x1,1.0)
                value += v1.pow(2).mean() * 0.5
                return x1 - v1, value
                
            for i in range(Nt):
                x1 = odeint(self.net, x1, 1-i * tau, 1-(i+1) * tau)
            xresult = x1
        else:
            x0 = xx
            if running:
                v0 = self.net(x0,0.0)
                value += v0.pow(2).mean() * 0.5
                return x0 + v0, value
            
            for i in range(Nt):
                x0 = odeint(self.net, x0, i * tau, (i+1) * tau)
            xresult = x0
            
        return xresult
    
    def inv(self, x, Nt=5, running=False):
        return self.forward(x, Nt=Nt, running=running, inv=True)
        
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


def compute_GME_cost(T, x, more=False):
    Tx = T(x).view((x.shape[0],-1))
    x  = x.view((x.shape[0],-1))
    
    Txy = ((Tx.view((Tx.shape[0],1,Tx.shape[1]))-Tx.view((1,Tx.shape[0],Tx.shape[1])))**2).sum(2)
    xy  = (( x.view(( x.shape[0],1, x.shape[1]))- x.view((1, x.shape[0], x.shape[1])))**2).sum(2)
    
    ATxy = (1.0+Txy).log()
    Axy  = (1.0+xy ).log()
    loss = ((ATxy - Axy)**2).mean()
    
    if more == True:
        xy  = xy.detach().cpu().numpy()
        Txy = Txy.detach().cpu().numpy()
        xy  = xy[np.triu_indices(x.shape[0],1)]
        Txy = Txy[np.triu_indices(x.shape[0],1)]
        return loss, xy**0.5, Txy**0.5
    return loss




def iterate_FMOT(Tx,z,R,optR,phi,optphi,count_c=0, skip_c=5, grad_clip=1, rate=0.9, C_phi=100.0, phi_noise=1e-2):
    """perform a single iteration for FMOT method

    Args:
        Tx (torch vector): embedded data samples on zdim
        z (torch vector): random variable on zdim
        R (torch module): flow map from z to Tx
        optR (optim): optimizer for R
        phi (torch module): dual varaible for the pushforward constraint
        optphi (optim): optimizer for phi
        count_c (int, optional): this will be used to perform minimax. Defaults to 0.
    """
    device = Tx.device
    
    optR.zero_grad()
    loss_R_total = R.compute_velocity_cost(Tx,size=5) + R.compute_velocity_R_cost(z, size=5) * 0.5
    loss_R_total.backward()
    torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)  # new
    optR.step()

    if count_c % skip_c == 0:
        c = C_phi * rate**(count_c/skip_c)
        # c = 30.0/(1+(count_c/skip_c)%300)+ 1e-3
        # c = 20 * (1 + np.log(1+epoch)) ** (- count_c/skip_c)
        
        optR.zero_grad()
        Rz, running_cost = R(z, running=True)
        loss_Running    = running_cost.mean()
        loss_constraint = (phi(Tx).detach()).mean() - phi(Rz).mean() 
        loss_R_total = (loss_Running + loss_constraint)
        loss_R_total.backward()
        torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)  # new
        optR.step()
    
        Rz, _     = R(z, running=True)
        
        optphi.zero_grad()
        phiRz = phi(Rz.detach())
        phiTx = phi(Tx.detach())
        loss_phi = phiRz.mean() - phiTx.mean()
        phiRz2 = phiRz.pow(2).mean()
        phiTx2 = phiTx.pow(2).mean()
        
        # Tx_noise = torch.randn(Tx.shape, device=device) * 2
        Tx_noise = (Tx + torch.randn(Tx.shape, device=device) * phi_noise).detach()
        Tx_noise = Tx_noise.requires_grad_(True)
        phiTx = phi(Tx_noise)
        grad_out  = torch.ones((Tx.shape[0],1), device=device, requires_grad=False)
        grad      = autograd.grad( phiTx, Tx_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.pow(2).mean(1).pow(1).mean()
        loss_phi += (grad_norm + 0.1 * phiTx2) * c
        
        Rz_noise = (Rz + torch.randn(Rz.shape, device=device) * phi_noise).detach()
        Rz_noise = Rz_noise.requires_grad_(True)
        phiRz  = phi(Rz_noise)
        grad_out  = torch.ones((Rz.shape[0],1), device=device, requires_grad=False)
        grad      = autograd.grad( phiRz, Rz_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.pow(2).mean(1).pow(1).mean()
        loss_phi += (grad_norm + 0.1 * phiRz2) * c
        
        loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), grad_clip)  # new
        optphi.step()
        
def laplacian(x,phi):
    xy    = (( x.view(( x.shape[0],1, x.shape[1]))- x.view((1, x.shape[0], x.shape[1])))**2).mean(2)
    phixy = ((phi.view((phi.shape[0],1,phi.shape[1]))-phi.view((1,phi.shape[0],phi.shape[1])))**2).mean(2)
    Axy   = 1.0/(1.0+xy )
    return (Axy * phixy).mean()
    

def iterate_FMOT_v2(Tx,z,R,optR,phi,phi_copy,optphi,count_c=0, skip_c=5, grad_clip=1, rate=0.9, C_phi=100.0, phi_noise=1e-2):
    """perform a single iteration for FMOT method
        phi is different from the v1
        phi^{k+1} = phi^{k} + sigma * nabla(J(phi^k) + |nabla(phi^{k} - phi^{k-1}|^2))

    Args:
        Tx (torch vector): embedded data samples on zdim
        z (torch vector): random variable on zdim
        R (torch module): flow map from z to Tx
        optR (optim): optimizer for R
        phi (torch module): dual varaible for the pushforward constraint
        optphi (optim): optimizer for phi
        count_c (int, optional): this will be used to perform minimax. Defaults to 0.
    """
    
    optR.zero_grad()
    loss_R_total = R.compute_velocity_cost(Tx,size=5) + R.compute_velocity_R_cost(z, size=5) * 0.5
    loss_R_total.backward()
    torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)  # new
    optR.step()

    if count_c % skip_c == 0:
        c = 50
        # c = 5.0
        
        optR.zero_grad()
        Rz, running_cost = R(z, running=True)
        loss_Running    = running_cost.mean()
        loss_constraint = (phi(Tx).detach()).mean() - phi(Rz).mean() 
        loss_R_total = (loss_Running + loss_constraint)
        loss_R_total.backward()
        torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)  # new
        optR.step()
    
        Rz, _     = R(z, running=True)
        Rz = Rz.detach()
            
        optphi.zero_grad()
        phiRz = phi(Rz.detach())
        phiTx = phi(Tx.detach())
        loss_phi = phiRz.mean() - phiTx.mean()
        
        # Tx_noise = torch.randn(Tx.shape, device=device) * 2
        # Tx_noise = (Tx + torch.randn(Tx.shape, device=device) * phi_noise).detach()
        # Tx_noise = Tx
        # Tx_noise = Tx_noise.requires_grad_(True)
        # phiTx = phi(Tx_noise) - phi_copy(Tx_noise)
        # grad_out  = torch.ones((Tx.shape[0],1), device=device, requires_grad=False)
        # grad      = autograd.grad( phiTx, Tx_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        # grad = grad.view(grad.shape[0], -1)
        # grad_norm = grad.pow(2).mean(1).pow(1).mean()
        # # loss_phi += (grad_norm + 0.01 * phiTx.pow(2).mean()) * c
        # loss_phi += (grad_norm) * c
        
        # Rz_noise = (Rz + torch.randn(Rz.shape, device=device) * phi_noise).detach()
        # Rz_noise = Rz
        # Rz_noise = Rz_noise.requires_grad_(True)
        # phiRz  = phi(Rz_noise) - phi_copy(Rz_noise)
        # grad_out  = torch.ones((Rz.shape[0],1), device=device, requires_grad=False)
        # grad      = autograd.grad( phiRz, Rz_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        # grad = grad.view(grad.shape[0], -1)
        # grad_norm = grad.pow(2).mean(1).pow(1).mean()
        # # loss_phi += (grad_norm + 0.01 * phiRz.pow(2).mean()) * c
        # loss_phi += (grad_norm) * c
        
        # Tx_noise = Tx
        # Tx_noise = Tx_noise.requires_grad_(True)
        
        # phiTx = phi(Tx_noise) - phi_copy(Tx_noise)
        # grad_out  = torch.ones((Tx.shape[0],1), device=device, requires_grad=False)
        # grad      = autograd.grad( phiTx, Tx_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        # grad = grad.view(grad.shape[0], -1)
        # grad_norm = grad.pow(2).mean(1).pow(1).mean()
        # # loss_phi += (grad_norm + 0.01 * phiTx.pow(2).mean()) * c
        # loss_phi += (grad_norm) * c
        
        Tx_Rz = torch.concat((Tx,Rz),0)
        phiRz  = phi(Tx_Rz) - phi_copy(Tx_Rz)
        loss_phi += laplacian(Tx_Rz, Tx_Rz) * c
        
        loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), grad_clip)  # new
        optphi.step()
        


def iterate_FMOT2(Tx,z,R,R2,optR,optR2,phi,optphi,FM_cost, count_c=0, skip_c=5, grad_clip=1, rate=0.9, C_phi=10.0):
    """perform a single iteration for FMOT method

    Args:
        Tx (torch vector): embedded data samples on zdim
        z (torch vector): random variable on zdim
        R (torch module): flow map from z to Tx
        optR (optim): optimizer for R
        phi (torch module): dual varaible for the pushforward constraint
        optphi (optim): optimizer for phi
        count_c (int, optional): this will be used to perform minimax. Defaults to 0.
    """
    device = Tx.device
    
    optR.zero_grad()
    # loss = R.compute_velocity_cost(Tx,size=1)
    t, xt, ut = FM_cost.sample_location_and_conditional_flow(z, Tx)
    vt = R.net(xt, t)
    loss = (vt - ut).pow(2).mean() + R.compute_velocity_R_cost2(z, R2, size=1) * 0.5
    loss.backward()
    torch.nn.utils.clip_grad_norm_(R.parameters(), grad_clip)  # new
    optR.step()

    if count_c % skip_c == 0:
        c = C_phi * rate**(count_c/skip_c)
        # c = 30.0/(1+epoch)+ 1e-3
        # c = 20 * (1 + np.log(1+epoch)) ** (- count_c/skip_c)
        
        optR2.zero_grad()
        Rz, running_cost = R2(z, running=True)
        loss_Running    = running_cost.mean() + R2.compute_velocity_cost(Tx,size=1,t=1.0)
        loss_constraint = (phi(Tx).detach()).mean() - phi(Rz).mean() 
        loss_R_total = (loss_Running + loss_constraint)
        loss_R_total.backward()
        torch.nn.utils.clip_grad_norm_(R2.parameters(), grad_clip)  # new
        optR2.step()
    
        Rz, _     = R2(z, running=True)
        
        optphi.zero_grad()
        phiRz = phi(Rz.detach())
        phiTx = phi(Tx.detach())
        loss_phi = phiRz.mean() - phiTx.mean()
        phiRz2 = phiRz.pow(2).mean()
        phiTx2 = phiTx.pow(2).mean()
        
        # Tx_noise = torch.randn(Tx.shape, device=device) * 2
        Tx_noise = (Tx + torch.randn(Tx.shape, device=device) * 3e-1).detach()
        Tx_noise = Tx_noise.requires_grad_(True)
        phiTx = phi(Tx_noise)
        grad_out  = torch.ones((Tx.shape[0],1), device=device, requires_grad=False)
        grad      = autograd.grad( phiTx, Tx_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.pow(2).mean(1).pow(1).mean()
        loss_phi += (grad_norm + 0.1 * phiTx2) * c
        
        Rz_noise = (Rz + torch.randn(Rz.shape, device=device) * 3e-1).detach()
        Rz_noise = Rz_noise.requires_grad_(True)
        phiRz  = phi(Rz_noise)
        grad_out  = torch.ones((Rz.shape[0],1), device=device, requires_grad=False)
        grad      = autograd.grad( phiRz, Rz_noise, grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.pow(2).mean(1).pow(1).mean()
        loss_phi += (grad_norm + 0.1 * phiRz2) * c
        
        loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), grad_clip)  # new
        optphi.step()
        
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
    def __init__(self, batch_size=100, sample_size=1000, plot_freq=200, MAX_OUTER_ITER=500, zdim=100):
        self.batch_size  = batch_size 
        self.sample_size = sample_size
        self.plot_freq   = plot_freq
        self.MAX_OUTER_ITER = MAX_OUTER_ITER
        self.zdim = zdim
        
      
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
        
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim+1, dd),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(dd, dd),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(dd, dd),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(dd, dd),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(dd, output_dim),
        # )

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
    
