#!/usr/bin/python
""" 
    Running simple sinkhorn algorithm.
"""

import argparse
import sys
import tqdm
import time
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
parser.add_argument('--num_epochs', type=int, default=10000)
# arguments to choose dataset (mnist, cifar, cifar-gray etc.)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--plot_every', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--saving', type=str, default="0")
parser.add_argument('--fig', type=str, default="0")
parser.add_argument('--data', type=str, default="celeb")
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--tqdm', type=int, default=0)

# Training T or phi
parser.add_argument('--load_T',    type=int, default=0)
parser.add_argument('--save_T',    type=int, default=0)

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
    xshape = (1,img_size,img_size)
    
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    # Load the dataset from file and apply transformations
    dataset_nn = datasets.MNIST(data_dir,download=True,transform=transform,)
    PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100)
elif data_str == 'cifar':
    from transportmodules.transportsCifar import *
    data_dir = '../data/cifar'
    zdim = 100
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

    zdim = 500
    # scale = 0.7
    scale = 300

    img_size = 64
    xshape = (3,img_size,img_size)
    
    
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # Load the dataset from file and apply transformations
    dataset_nn = CelebADataset(data_dir, transform)
    PARAM = Parameters(batch_size=100, sample_size=1000, plot_freq=100)

    if args.saving == "1":
        zdim = 100
        scale = 1.0
elif data_str == 'celebahq':
    from transportmodules.transportsCelebHQ import *
    data_dir = '../data/celebahq/celeba_hq_256'
    zdim = 500
    img_size = 256
    scale = 1.0
    xshape = (3,img_size,img_size)
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset_nn = CustomCelebAHQ(data_dir,transform=transform,)
    PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=100)

    if args.saving == "1":
        zdim = 100
        scale = 1.0
else:
    print("wrong input for data")
    sys.exit()

print(f'zdim: {zdim} scale: {scale} data: {args.data} saving: {args.saving}')

def plot_and_save(T, x_val, total_iterations, figure_count, elapsed_time_T, save_data_path, save_fig_path, loss_arr_T):
    with torch.no_grad():
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        x = x_val[:100]
        loss = compute_GME_cost(T,x)
        loss_arr_T.append(loss.item())
        ax.plot(loss_arr_T)
        ax.set_yscale('log')
        ax.set_title(f'iterations:{total_iterations} figure_count:{figure_count_T}\n time: {elapsed_time_T:9.2e}')
        fig.savefig(f'{save_fig_path}/T-{total_iterations}.png')
        plt.close('all')

# Number of workers for the dataloader
num_workers = 0 if cuda else 2
# Whether to put fetched data tensors to pinned memory
pin_memory = True if cuda else False
    
save_fig_path  = f'fig_{data_str}_{args.saving}_{args.fig}' ;os.makedirs(save_fig_path,  exist_ok=True);print(f"saving images in {save_fig_path}")
save_data_path = f'data_{data_str}_{args.saving}';os.makedirs(save_data_path, exist_ok=True);print(f"saving data in {save_data_path}")

# %%
dataloader = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
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
dataloader_valid = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.sample_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

figure_count_T = 0
elapsed_time_T = 0
pbar = tqdm.tqdm(range(1_000_000))
figure_count = 0
start_time_T = time.time()
loss_arr_T = []
for total_iterations in pbar:    
    imgs, _ = next(iter(dataloader))
    x = imgs.to(device).detach()
    optT.zero_grad()
    loss = compute_GME_cost(T,x)
    loss.backward()
    optT.step()

    pbar.set_description(f'GME loss: {loss.item():9.2e}')

    # save the model and plot every 1000 iterations
    if total_iterations%1000==0:
        elapsed_time_T += time.time() - start_time_T
        figure_count += 1
        if args.save_T == 1:
            torch.save(T.state_dict(), f"{save_data_path}/T.pt")
        plot_and_save(T, x_val, total_iterations, figure_count, elapsed_time_T, save_data_path, save_fig_path, loss_arr_T)
        start_time_T = time.time()