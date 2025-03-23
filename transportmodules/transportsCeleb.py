"""
This Python file defines PyTorch modules for the encoder, decoder, and flow map 
used in a flow-based generative model. 
These modules are tailored for image datasets, specifically for 
the CelebA dataset, and can be adapted to different input and output shapes. 
"""

import torch
import torch.nn as nn
import numpy as np
from torchcfm.models.unet.unet import UNetModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class NetSingle(torch.nn.Module):
    def __init__(self, xdim = 100, zdim = 1, Nt = 10):
        super(NetSingle, self).__init__()
        self.zdim = zdim
        
        self.Nt = Nt
        self.n_frequencies = 5
    
        self.in_shape = (4, 8, 8)

        self.lin_in = nn.Sequential(
            nn.Linear(xdim, np.prod(self.in_shape)),
            nn.Unflatten(1, (self.in_shape)),
        )
        self.lin_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.in_shape), xdim),
        )
        
        self.model = UNetModelWrapper(
            dim=self.in_shape,
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[2,2,2],
            num_heads=4,
            num_head_channels=16,
            attention_resolutions="4",
            dropout=0.1,
        ).to(device)  # new dropout + bs of 128
    
    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t[..., None]
        aa = torch.cat((t.cos(), t.sin()), dim=-1)
        return aa.view((aa.shape[0],-1))
    
    def net(self, x: torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        encoded_t = self.time_encoder(torch.ones((x.shape[0],1),device=device)*t)
        # print(x.shape,encoded_t.shape)
        # return self.model(torch.cat((x, encoded_t), 1))
        
        encoded_x = self.lin_in(x)
        return self.lin_out(self.model(encoded_t, encoded_x))
        
    def forward(self, x:torch.Tensor, t=None, i=0):
        if t == None:
            t = i/self.Nt if i>=0 else 1 - (-i-1)/self.Nt
        return self.net(x,t)

class TransportT(torch.nn.Module):
    def __init__(self, input_shape = [3,32,32], zdim = 100):
        super(TransportT, self).__init__()

        self.input_shape = input_shape
        self.zdim = zdim
        n_channels = input_shape[0]
        img_size   = input_shape[1]
        conv_channel = 32
        
        def discriminator_block(in_filters, out_filters, downscale=True, bn=True):
            stride = 2 if downscale else 1
            block = [nn.Conv2d(in_filters, out_filters, 5, stride, 2), 
                     nn.LeakyReLU(0.2,True), 
                    #  nn.Dropout2d(0.25)
                    ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.main = nn.Sequential(
            *discriminator_block(n_channels, conv_channel),
            *discriminator_block(conv_channel, conv_channel*2,  bn=False),
            *discriminator_block(conv_channel*2, conv_channel*4),
            *discriminator_block(conv_channel*4, conv_channel*8),
            *discriminator_block(conv_channel*8, conv_channel*16),
        )

        self.output_main   = conv_channel*8*(img_size//16)**2

        self.fc_T = nn.Sequential(
            nn.Conv2d(conv_channel*16, zdim, 2, 1, 0),
        )
        
        self.scale= 1.0

    def forward(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main(x)
        x = self.fc_T(x).view(x.shape[0],self.zdim)
        return x * self.scale
    
    def update_mean_(self, mean_arr):
        self.mean = Tensor(mean_arr).view((1,self.zdim))
    def update_scale_(self, scale):
        self.scale = scale
    
class TransportG(torch.nn.Module):
    def __init__(self, zdim = 100, output_shape = [1,32,32]):
        super(TransportG, self).__init__()

        self.zdim = zdim
        self.output_shape = output_shape
        self.conv_shape = (256, 16, 16)

        self.R = nn.Sequential(
            nn.Linear(zdim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, np.prod(self.conv_shape)),
        )

        self.S = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 11, stride=1, padding=5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 11, stride=1, padding=5),
            # state size. 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.R(x)
        x = x.view(x.shape[0], *self.conv_shape)
        x = self.S(x)
        return x
  