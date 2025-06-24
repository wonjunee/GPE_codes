#!/usr/bin/python
"""
This script trains the GPE decoder using the specified dataset after training the encoder.
Supported datasets include: MNIST, CIFAR-10, CelebA, and CelebA-HQ.

To use a different dataset, modify the data loader section accordingly.
"""


import argparse
import sys
import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from utils.functions import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9" 

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--num_iter', type=int, default=1_000_000)
# arguments to choose dataset (mnist, cifar, cifar-gray etc.)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--plot_every', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--saving', type=str, default="0")
parser.add_argument('--fig', type=str, default="0")
parser.add_argument('--data', type=str, default="celeb")
parser.add_argument('--S_parallel', type=bool, default=True)


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
    scale = 0.3
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
    scale = 0.1
    img_size = 256
    xshape = (3,img_size,img_size)
    # Transformations to be applied to each individual image sample
    transform=transforms.Compose([ transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # Load the dataset from file and apply transformations
    dataset_nn = CustomCelebAHQ(data_dir,transform=transform,)
    PARAM = Parameters(batch_size=50, sample_size=1000, plot_freq=100)
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

# %% loading pretrained encoder
dataloader = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

T = TransportT(input_shape=xshape, zdim=zdim).to(device)
T.load_state_dict(torch.load(f'{save_data_path}/T.pt', map_location=torch.device(device), weights_only=True))
print(f"Encoder loaded.")

# %% Training the decoder
S = TransportG(output_shape=xshape,zdim=zdim).to(device)
S_PARARELL = args.S_parallel

if S_PARARELL:
    S = torch.nn.DataParallel(S).to(device)    

lr = 1e-5
b1 = 0.5
b2 = 0.999
optS = torch.optim.Adam(S.parameters(), lr=1e-4, betas=(b1, b2))
pbar = tqdm.tqdm(range(args.num_iter))

for total_iterations in pbar:    
    imgs, _ = next(iter(dataloader))
    x = imgs.to(device).detach()
    Tx = T(x).detach()
    optS.zero_grad()
    STx = S(Tx.detach())
    loss = F.mse_loss(x, STx)
    loss.backward()
    optS.step()

    recons_loss = loss.item()
    pbar.set_description(f'recons loss: {recons_loss:9.2e}')

    # Save the module T every 1000 iterations
    if total_iterations % 1000 == 0:
        save_path = f'{save_data_path}/S.pt'
        torch.save(S.state_dict(), save_path)
        print(f'Model saved at iteration {total_iterations} to {save_path}')
