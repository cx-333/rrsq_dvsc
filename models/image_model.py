# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F
from layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2


g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )
        self.enc_feature= None

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        return self.forward_torch(out, quant_step)

    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        self.enc_feature = out
        out = out * quant_step
        return self.enc_2(out)
    

class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)
        self.dec_feature = None
       

    def forward(self, x, quant_step):
        return self.forward_torch(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        self.dec_feature = out
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class IntraModel(nn.Module):
    def __init__(self, N=256, z_channel=128):
        super().__init__()

        self.enc = IntraEncoder(N)
        self.dec = IntraDecoder(N)

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))

    @staticmethod
    def get_qp_num():
        return 64

    def forward(self, x, qp):
        """ ✅ perfect
        Forward pass for DMCI model during training.
        Args:
            x: Input image tensor [B, C, H, W]
            qp: Quantization parameter index
        Returns:
            dict: Contains reconstructed image and rate information
        """
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        # Encode input to latent representation
        y = self.enc(x, curr_q_enc)
        y_hat = torch.round(y)
        
        # Decode to reconstructed image
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        
        
        return {
            "x_hat": x_hat,
        }
