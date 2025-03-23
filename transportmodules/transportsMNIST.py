"""
This Python file defines PyTorch modules for the encoder, decoder, and flow map 
used in a flow-based generative model. 
These modules are tailored for image datasets, specifically for 
the MNIST dataset, and can be adapted to different input and output shapes. 
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
        self.L  = Nt+1
        self.n_frequencies = 4
        
        if True:
            self.in_shape = (1, 8, 8)

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
                num_channels=64,
                channel_mult=[2,2,2,2],
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
    
    def net(self, x, t):
        encoded_t = self.time_encoder(torch.ones((x.shape[0],1),device=device)*t)
        encoded_x = self.lin_in(x)
        return self.lin_out(self.model(encoded_t, encoded_x))
        
        
    def forward(self, x, t=None, i=0):
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
        )

        self.output_main   = conv_channel*4*(img_size//8)**2

        self.fc_T = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_main, zdim)
        )
        
        self.scale= 1.0

    def forward(self, x):
        x = self.main(x)
        x = self.fc_T(x).view(x.shape[0],self.zdim)
        return x
    
class TransportG(torch.nn.Module):
    def __init__(self, zdim = 100, output_shape = [1,32,32]):
        super(TransportG, self).__init__()

        self.zdim = zdim
        self.output_shape = output_shape

        self.S = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(zdim, 128, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32 x 32
            nn.ConvTranspose2d(32, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, output_shape[0], 5, stride=1, padding=2),
            # state size. 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.S(x)
        return x