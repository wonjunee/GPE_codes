#!/usr/bin/python
"""
This script trains the conditional flow matching model after
training GPE encoder and decoder.
Supported datasets include: MNIST, CIFAR-10, CelebA, and CelebA-HQ.

To use a different dataset, modify the data loader section accordingly.
"""

import argparse
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

import copy

# import zuko.utils as zukoutils
# from torchdyn.core import NeuralODE
from utils.functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9" 

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--num_iter', type=int, default=100_000_000)
# arguments to choose dataset (mnist, cifar, cifar-gray etc.)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--plot_every', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--saving', type=str, default="0")
parser.add_argument('--fig', type=str, default="0")
parser.add_argument('--data', type=str, default="celeb")
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--S_parallel', type=bool, default=True)


args = parser.parse_args()
print(args)

# system preferences
torch.set_default_dtype(torch.float)
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(2)

data_str = args.data.lower()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    num_gpus = torch.cuda.device_count()
    print(f"XXX {num_gpus} GPU available XXX")
else:
    print("XXX GPU is not available XXX")

if data_str == 'mnist':
    from transportmodules.transportsMNIST import *
    data_dir = '../data/mnist'
    zdim = 30
    img_size = 32
    scale = 1.0
    xshape = (1,img_size,img_size)
    
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    # Load the dataset from file and apply transformations
    dataset_nn = datasets.MNIST(data_dir,download=True,transform=transform,)
    PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100)
elif data_str == 'cifar':
    from transportmodules.transportsCifar import *
    data_dir = '../data/cifar'
    zdim = 50
    scale = 1.0
    img_size = 32
    xshape = (3,img_size,img_size)
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # Define transformations for data augmentation
    transform = transforms.Compose(
                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
                )
    # Load the dataset from file and apply transformations
    dataset_nn = datasets.CIFAR10(data_dir,download=False,transform=transform,)
    PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100)
elif data_str == 'celeb':
    from transportmodules.transportsCeleb import *
    data_dir = '../data/CelebA/img_align_celeba'

    zdim = 100
    scale = 1.0

    img_size = 64
    xshape = (3,img_size,img_size)
    
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # Load the dataset from file and apply transformations
    dataset_nn = CelebADataset(data_dir, transform)
    PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100)

elif data_str == 'celebahq':
    from transportmodules.transportsCelebHQ import *
    data_dir = '../data/celebahq/celeba_hq_256'
    zdim = 100
    img_size = 256
    scale = 1.0
    xshape = (3,img_size,img_size)
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset_nn = CustomCelebAHQ(data_dir,transform=transform,)
    PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=100)
else:
    print("wrong input for data")
    sys.exit()

# Number of workers for the dataloader
num_workers = 0 if cuda else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if cuda else False
    
save_fig_path  = f'fig_{data_str}_{args.saving}_{args.fig}' ;os.makedirs(save_fig_path,  exist_ok=True);print(f"saving images in {save_fig_path}")
save_data_path = f'data_{data_str}_{args.saving}';os.makedirs(save_data_path, exist_ok=True);print(f"saving data in {save_data_path}")

# %%
dataloader = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.sample_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

# getting the validation set
x_val = []
len_tmp = 0
for img, _ in dataloader:
    x_val.append(img)
    len_tmp += len(img)
    if len_tmp >= PARAM.sample_size:
        break
x_val = torch.concat((x_val), 0)
x_val = x_val.to(device)

lr = 1e-5
b1 = 0.5
b2 = 0.999

T = TransportT(input_shape=xshape, zdim=zdim).to(device)
optT = torch.optim.Adam(T.parameters(), lr=1e-4, betas=(b1, b2))

# %%
dataloader = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
x_val = []
len_tmp = 0
while len_tmp < PARAM.sample_size:
    img, _ = next(iter(dataloader))
    x_val.append(img)
    len_tmp += len(img)
x_val = torch.concat((x_val), 0)
x_val = x_val.to(device)

lr = 1e-5
b1 = 0.5
b2 = 0.999

T = TransportT(input_shape=xshape, zdim=zdim).to(device)
try:
    T.load_state_dict(torch.load(f'{save_data_path}/T.pt',map_location=torch.device(device), weights_only=True) )
except:
    T = torch.nn.DataParallel(T).to(device)    
    T.load_state_dict(torch.load(f'{save_data_path}/T.pt',map_location=torch.device(device), weights_only=True) )
T_scale = lambda x: T(x) * scale

# lading S
S = TransportG(output_shape=xshape,zdim=zdim).to(device)
S_PARARELL = args.S_parallel
try:
    S.load_state_dict(torch.load(f'{save_data_path}/S.pt',map_location=torch.device(device), weights_only=True) )
except:
    S = torch.nn.DataParallel(S).to(device)
    S.load_state_dict(torch.load(f'{save_data_path}/S.pt',map_location=torch.device(device), weights_only=True) )

# %%
print("XXX Training R XXX")
R    = NetSingle(xdim=zdim,zdim=zdim).to(device)
optR = torch.optim.Adam(R.parameters(), lr=2e-5, betas=(b1, b2))

pbar = tqdm.tqdm(range(args.num_iter))
figure_count = 0

for iteration in pbar:
    imgs, _ = next(iter(dataloader))
    x = imgs.to(device).detach()
    Tx = T_scale(x).detach()
    z = get_latent_samples(shape=(PARAM.batch_size,zdim),device=device).detach()

    # Compute the CFM cost
    optR.zero_grad()
    t = torch.rand(x.shape[0],1,device=Tx.device).detach()
    xx = (1-t) * z + t * Tx
    u1 = Tx - z
    loss = (R(xx.detach(), t) - u1.detach()).pow(2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(R.parameters(), args.grad_clip)
    optR.step()
    
    pbar.set_description(f'Total iterations: {iteration}')
    
    # Save the module every 10_000 iteration
    if iteration % 10_000 == 0:
        torch.save(R.state_dict(), f"{save_data_path}/R-{iteration}.pt")

    
    # ----- Plotting -----
    if iteration % 100_000 == 0:
        pbar.set_description(f'Plotting ...')
        
        Nt_plot = 10
        with torch.no_grad():
            # Create a figure with subplots
            fig, (ax_x, ax_STx, ax_SRz) = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
            
            x = x_val[:50]
            
            # getting embedded and decoded points
            Tx  = T(x).detach()
            STx = S(Tx).detach()
            
            # normalizing
            x   = (x+1)/2.0
            STx = (STx+1)/2.0

            nrow = 6
            ncol = 6
            
            ax_x.imshow(get_n_by_n_images(x,nrow,ncol))
            ax_x.set_title(f'It: {iteration:9.2e}')
            ax_x.axis('off')
            ax_STx.imshow(get_n_by_n_images(STx,nrow,ncol))
            ax_STx.set_title('STx')
            ax_STx.axis('off')
            
            ### generation
            z_val = get_latent_samples((nrow*ncol, zdim), device=device)
            Rz = pushforward(R, z_val, Nt=Nt_plot).detach() / scale
            SRz = S(Rz).detach()
            ax_SRz.imshow(get_n_by_n_images(SRz,nrow,ncol))
            ax_SRz.set_title(f'{t}')
            ax_SRz.axis('off')
            
            del STx, SRz, Rz
            
            filename = f'{save_fig_path}/G-{figure_count}.png'
            plt.savefig(filename)
            plt.close('all')
            
        figure_count += 1