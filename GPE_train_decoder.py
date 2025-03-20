#!/usr/bin/python
""" 
    Running simple sinkhorn algorithm.
"""

import argparse
import pickle
import sys
import tqdm
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from torchvision import datasets
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

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
parser.add_argument('--train_T',   type=int, default=0) # train T if 0 and don't train T if else
parser.add_argument('--train_S',   type=int, default=0) # train T if 0 and don't train T if else
parser.add_argument('--load_T',    type=int, default=0)
parser.add_argument('--save_T',    type=int, default=0)
parser.add_argument('--load_S',    type=int, default=0)
parser.add_argument('--save_S',    type=int, default=0)

args = parser.parse_args()
print(args)

# system preferences
torch.set_default_dtype(torch.float)
seed = np.random.randint(100)
# seed = np.random.randint(101)
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
elif data_str == 'stl10':
    from transportmodules.transportsSTL10 import *
    data_dir = '../data/stl10'
    img_size = 96
    xshape = (3,img_size,img_size)
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)),  # Random crop without padding
        transforms.Resize(img_size), 
        transforms.CenterCrop(img_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset_nn = datasets.STL10(root=data_dir, split='train', download=True, transform=transform)
    dataset_nn = dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn + dataset_nn
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
elif data_str == 'lsun':
    from transportmodules.transportsCeleb import *
    data_dir = '../data/lsun'
    zdim = 500
    img_size = 128
    xshape = (3,img_size,img_size)
    scale = 1.0
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset_nn = datasets.LSUN(data_dir,classes=['bedroom_train'],transform=transform,)
    PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=500)
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

img_skip = img_size // 32

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
if args.load_T == 1:
    T.load_state_dict(torch.load(f'{save_data_path}/T.pt', map_location=torch.device(device), weights_only=True))

optT = torch.optim.Adam(T.parameters(), lr=1e-4, betas=(b1, b2))

# if args.load_T == 1:
create_xy_plot(x_val, T, zdim, dataloader, save_fig_path)

# %%
dataloader_valid = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.sample_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

if args.train_T != 0:
    if args.load_T == 1:
        T.load_state_dict(torch.load(f'{save_data_path}/T.pt',map_location=torch.device(device), weights_only=True) )  
    loss_arr_T= []
    
    error_nums = [0.5, 0.03, 0.004, 0.0018, 0.0009, 0.0007]
    error_dic = {f'{t}': True for t in error_nums}
    
    figure_count_T = 0
    elapsed_time_T = 0
    pbar = tqdm.tqdm(range(1_000_000))
    figure_count = 0
    start_time_T = time.time()
    for total_iterations in pbar:    
        imgs, _ = next(iter(dataloader))
        x = imgs.to(device).detach()
        optT.zero_grad()
        loss = compute_GME_cost(T,x)
        loss.backward()
        optT.step()

        pbar.set_description(f'GME loss: {loss.item():9.2e}')

        if total_iterations%100==0:
            elapsed_time_T += time.time() - start_time_T
            with torch.no_grad():
                figure_count += 1
                if args.save_T == 1:
                    torch.save(T.state_dict(), f"{save_data_path}/T.pt")
                fig,ax = plt.subplots(1,1,figsize=(5,5))
                x = x_val[:100]
                loss = compute_GME_cost(T,x)
                loss_arr_T.append(loss.item())
                ax.plot(loss_arr_T)
                ax.set_yscale('log')
                ax.set_title(f'iterations:{total_iterations} figure_count:{figure_count_T}\n time: {elapsed_time_T:9.2e}')
                plt.savefig(f'{save_fig_path}/0000Txstatus.png')
                plt.savefig(f'{save_fig_path}/T-{total_iterations}.png')
                plt.close('all')
                
                for t0 in error_nums:
                    if error_dic[f'{t0}'] == True and loss < t0 and total_iterations > 100:
                        pbar.write(f"Training done!! t0: {t0:9.2e} GME loss: {loss.item():9.2e} time: {elapsed_time_T/60:9.2e} min  Iterations: {total_iterations}")
                        error_dic[f'{t0}'] = False
                        torch.save(T.state_dict(), f"{save_data_path}/T-{t0}.pt")
                    
            start_time_T = time.time()

try:
    Tname = 'T.pt'
    if args.data == 'celeb':
        Tname = 'T-0.004.pt'
        # Tname = 'T-0.0007.pt'
        # Tname = 'T.pt'
    elif args.data == 'celebahq':
        # Tname = 'T-0.03.pt'
        # Tname = 'T.pt'
        Tname = 'T-0.004.pt'
    Tname = 'T.pt'
    T.load_state_dict(torch.load(f'{save_data_path}/{Tname}',map_location=torch.device(device), weights_only=True) )
    print(f"Loading {Tname}")
except:
    T = torch.nn.DataParallel(T).to(device)    
    T.load_state_dict(torch.load(f'{save_data_path}/{Tname}',map_location=torch.device(device), weights_only=True) )


S = TransportG(output_shape=xshape,zdim=zdim).to(device)
S_PARARELL = True

if args.train_S != 0:
    
    # training S    
    if args.load_S == 1:
        S.load_state_dict(torch.load(f'{save_data_path}/S.pt',map_location=torch.device(device), weights_only=True) )
    if S_PARARELL:
        S = torch.nn.DataParallel(S).to(device)    
    optS = torch.optim.Adam(S.parameters(), lr=1e-4, betas=(b1, b2))
    pbar = tqdm.tqdm(range(1_000_000))
    total_elapsed_time = 0
    total_iterations = 0
    recons_loss_arr =[]
    error_dic = {'0.025': True, '0.05': True, '0.1': True, '0.0225':True, '0.015': True, '0.012': True, '0.01': True}
    figure_count = 0
    start_time = time.time()
    for total_iterations in pbar:    
        imgs, _ = next(iter(dataloader))
        x = imgs.to(device).detach()
        Tx = T(x).detach()
        optS.zero_grad()
        Tx_noise = Tx #+ torch.randn(Tx.shape, device=device) * eps_Tx
        STx = S(Tx_noise.detach())
        loss = F.mse_loss(x, STx)
        loss.backward()
        optS.step()
        recons_loss = loss.item()

        for er in error_dic:
            er_float = float(er)
            if recons_loss < er_float and error_dic['0.1'] == True:
                total_elapsed_time_tmp = total_elapsed_time + time.time() - start_time
                error_dic[er] = False
                pbar.write(f'{er}. recons_loss: {recons_loss:9.2e} total_iterations: {total_iterations} Time: {total_elapsed_time_tmp} sec!')    

        recons_loss_arr.append(recons_loss)
        pbar.set_description(f'recons loss: {recons_loss:9.2e}')
        
        # ----- Plotting -----
        if total_iterations % 100 == 0 or total_iterations==1:
            
            elapsed_time = time.time() - start_time
            total_elapsed_time += elapsed_time

            pbar.write(f'total_iterations: {total_iterations} recons_loss: {recons_loss:9.2e} Time: {total_elapsed_time} sec!')
            
            with torch.no_grad():
                if args.save_S == 1:
                    if S_PARARELL:
                        torch.save(S.module.state_dict(), f"{save_data_path}/S.pt")
                    else:
                        torch.save(S.state_dict(), f"{save_data_path}/S.pt")
                # Create a figure and a gridspec layout
                # fig = plt.figure(figsize=(24, 10))  # Adjust figsize as needed
                fig = plt.figure(figsize=(50, 20))  # Adjust figsize as needed
                gs  = GridSpec(1, 3, width_ratios=[1, 1,1])  # 2 rows, 4 columns

                ax_x      = plt.subplot(gs[0, 0])
                ax_STx    = plt.subplot(gs[0, 1])
                ax_recon    = plt.subplot(gs[0, 2])
                
                count = 0
                x = x_val[:50]
                
                Tx  = T(x).detach()
                STx = S(Tx).detach()
                
                x   = (x+1)/2.0
                STx = (STx+1)/2.0

                if args.data == 'celebahq':
                    nrow = 3
                    ncol = 3
                else:
                    nrow = 6
                    ncol = 6
                
                ax_x.imshow(get_n_by_n_images(x,nrow,ncol))
                ax_x.set_title(f'total_iterations: {total_iterations}, time: {elapsed_time:0.2f} total: {total_elapsed_time:9.2e}')
                ax_x.axis('off')
                ax_STx.imshow(get_n_by_n_images(STx,nrow,ncol))
                ax_STx.set_title('STx')
                ax_STx.axis('off')
                
                ax_recon.plot(recons_loss_arr)
                ax_recon.set_title(f'recon loss: {recons_loss:9.2e}')
                fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95)

                filename = f'{save_fig_path}/G-{figure_count}.png'
                # pbar.write(f'saved in {filename} Reconstruction Loss: {recons_loss:9.2e}')
                plt.savefig(filename)
                plt.savefig(f'{save_fig_path}/0000S-status.png')
                plt.close('all')
            start_time = time.time()
            figure_count += 1