#!/usr/bin/python
"""
This script trains the GPE decoder using the specified dataset after training the encoder.
Supported datasets include: MNIST, CIFAR-10, CelebA, and CelebA-HQ.

To use a different dataset, modify the data loader section accordingly.
"""


import argparse
import sys
import time
import tqdm
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from pathlib import Path
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
parser.add_argument('--S_parallel', type=bool, default=False)


args = parser.parse_args()
print(args)

# system preferences
torch.set_default_dtype(torch.float)
np.random.seed(1)
torch.manual_seed(1)

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
    # data_dir = '../data/CelebA/img_align_celeba'
    data_dir = (Path.cwd().parent / "data" / "celebA" / "train" ).resolve() # path to celebA dataset (modify this path accordingly)
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
    data_dir = (Path.cwd().parent / "data" / "celeba_hq_256" ).resolve() # path to celebA dataset (modify this path accordingly)
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

# Whether to put fetched data tensors to pinned memory
pin_memory = True if cuda else False
# Number of workers for the dataloader
num_workers = num_gpus * 8 if cuda else 0
non_blocking = True if cuda else False
    
save_fig_path  = f'fig_{data_str}_{args.saving}_{args.fig}' ;os.makedirs(save_fig_path,  exist_ok=True);print(f"saving images in {save_fig_path}")
save_data_path = f'data_{data_str}_{args.saving}';os.makedirs(save_data_path, exist_ok=True);print(f"saving data in {save_data_path}")

# %% loading pretrained encoder
dataloader = torch.utils.data.DataLoader(dataset_nn, batch_size=PARAM.batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

# Collect 5000 samples from the validation dataloader
validation_samples = []
for imgs, _ in dataloader:
    validation_samples.append(imgs)
    if sum(batch.size(0) for batch in validation_samples) >= 5000:
        break

# Concatenate and trim to exactly 5000 samples, ensure it's a torch.Tensor
x_val = torch.cat(validation_samples, dim=0)[:5000]
x_val = torch.as_tensor(x_val).to(device, non_blocking=non_blocking).detach()
print(f"Validation set shape: {x_val.shape}")

T = TransportT(input_shape=xshape, zdim=zdim).to(device)
T.load_state_dict(torch.load(f'{save_data_path}/T.pth', map_location=torch.device(device), weights_only=True))
print(f"Encoder loaded.")

# %% Training the decoder
S = TransportG(output_shape=xshape,zdim=zdim).to(device)

if args.S_parallel:
    S = torch.nn.DataParallel(S).to(device)    

b1 = 0.5
b2 = 0.999
optS = torch.optim.Adam(S.parameters(), lr=1e-4, betas=(b1, b2))
pbar = tqdm.tqdm(range(args.num_iter))
recons_loss_arr = []

# --- timing setup ---
window = 1000                      # report every 1000 iterations
t0 = time.time()                   # start time for overall average
t_last = t0                        # start time for the current 1000-iter window

plot_freq = 1000

S.train()  # training mode
# If T is a module and is FROZEN, you can keep it in eval to avoid randomness
if isinstance(T, torch.nn.Module):
    T.eval()

total_iterations = 0
while total_iterations < args.num_iter:
    for imgs, _ in dataloader:   # <-- iterate properly over the loader
        x = imgs.to(device, non_blocking=non_blocking)
        Tx = T(x).detach()
        optS.zero_grad()
        STx = S(Tx)
        loss = (x - STx).pow(2).mean()
        loss.backward()
        optS.step()

        pbar.set_description(f"recons loss: {loss.item():9.2e}")

        # --- every 'plot_freq' iterations, do validation, save model, and plot ---
        if total_iterations % 1000 == 0:
            # --- timing block (prints every 'window' iters) ---
            t_now = time.time()
            dt_window = t_now - t_last
            avg_ms_window = (dt_window / window) * 1000.0
            dt_total = t_now - t0
            avg_ms_total = (dt_total / (total_iterations+1)) * 1000.0
            pbar.write(
                f"[Timing] {window} iters: "
                f"avg over last {window}: {avg_ms_window:.2f} ms/iter | "
                f"overall avg: {avg_ms_total:.2f} ms/iter"
            )

            # --- save every 1000 iterations ---
            torch.save(S.state_dict(), f"{save_data_path}/S.pth")

            # --- validation check and plotting ---
            S.eval()  # <-- IMPORTANT
            with torch.no_grad():
                # make sure x_val has SAME preprocessing as training
                Tx_val = T(x_val).detach()
                STx_val = S(Tx_val)

                val_loss = (x_val - STx_val).pow(2).mean()
                recons_loss_arr.append(val_loss.item())
                pbar.write(f"Validation loss at iteration {total_iterations}: {val_loss.item():9.2e}")
                xs = np.arange(len(recons_loss_arr)) * plot_freq
                np.savez(f"{save_data_path}/recons_loss_arr.npz", xs=xs, recons_loss_arr=np.array(recons_loss_arr))

                if False:
                    # If you normalized with mean=std=0.5, unnormalize; else skip
                    def denorm(z):
                        return (z * 0.5 + 0.5).clamp(0, 1)

                    n = 25
                    grid = make_grid(
                        torch.cat([denorm(x_val[:n]), denorm(STx_val[:n])], dim=0),
                        nrow=5
                    )

                    (Path(save_fig_path)).mkdir(parents=True, exist_ok=True)
                    save_image(grid, f"{save_fig_path}/recons_{total_iterations}.png")

                    # quick loss curve
                    plt.figure(figsize=(5,4))
                    xs = np.arange(len(recons_loss_arr)) * plot_freq
                    plt.plot(xs, recons_loss_arr)
                    plt.title(f"Reconstruction Loss (val)\nloss: {val_loss.item():9.2e}")
                    plt.xlabel("Iteration")
                    plt.ylabel("L2")
                    plt.yscale("log")
                    plt.tight_layout()
                    plt.savefig(f"{save_fig_path}/loss_{total_iterations}.png")
                    plt.close()

            S.train()  # back to train mode
            t_last = t_now # reset last-time for next window

        total_iterations += 1
        pbar.update(1)
        pbar.set_postfix({"iter": total_iterations})
