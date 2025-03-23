"""
This Python file defines PyTorch modules for the encoder, decoder, and flow map 
used in a flow-based generative model. 
These modules are tailored for image datasets, specifically for 
the CelebA-HQ (256x256) dataset, and can be adapted to different input and output shapes. 
"""

import torch
import torch.nn as nn
import numpy as np
from torchcfm.models.unet.unet import UNetModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class NetSingle(nn.Module):
    def __init__(self, xdim = 100, zdim = 1):
        super(NetSingle, self).__init__()

        self.lin_in = nn.Sequential(
            # Fully connected layer to change the input shape from (100, 51) to (100, 100)
            nn.Linear(xdim+1, 1024),           
            # Activation function (optional, e.g., ReLU)
            nn.ReLU(),
            # Reshape the output to (100, 1, 100)
            nn.Unflatten(dim=1, unflattened_size=(1, 1024))
        )

        self.lin_out = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            # Fully connected layer to change the input shape from (100, 51) to (100, 100)
            nn.Linear(1024, zdim),
        )

        in_channels = 1
        out_channels = 1

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Maxpooling
        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Final layer
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x:torch.Tensor, t=None):
        return self.net(x,t)
    
    def net(self, x, t):
        if isinstance(t, float):
            t = torch.ones(x.shape[0], 1, device=x.device) * t
        # Encoder path
        xt = torch.cat((x,t),1)
        lin_in1 = self.lin_in(xt)
        enc1 = self.enc1(lin_in1)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Final layer
        final_out =  self.final_conv(dec1).reshape((x.shape[0],-1))
        return self.lin_out(final_out)



class NetSingle2(torch.nn.Module):
    def __init__(self, xdim = 100, zdim = 1, Nt = 10):
        super(NetSingle2, self).__init__()
        self.zdim = zdim
        
        self.Nt = Nt
        self.n_frequencies = 5
    
        self.in_shape = (1, 32, 32)

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
            num_res_blocks=3,
            num_channels=256,
            channel_mult=[2,2,2],
            num_heads=4,
            num_head_channels=16,
            attention_resolutions="8",
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
            *discriminator_block(conv_channel*16, conv_channel*32),
            *discriminator_block(conv_channel*32, conv_channel*64),
        )

        self.fc_T = nn.Sequential(
            nn.Conv2d(conv_channel*64, zdim, 2, 1, 0),
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
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            # state size. 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.R(x)
        x = x.view(x.shape[0], *self.conv_shape)
        x = self.S(x)
        return x
    