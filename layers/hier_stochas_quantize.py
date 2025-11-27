# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F



class TopDownBlock(nn.Module):
    def __init__(self):
        super(TopDownBlock, self).__init__()
    
    def forward(self, z_cur, z_res, quantizer, codebook, log_param_q_scalar_q, flg_train, flg_quant_det):
        # Quantization
        param_q = log_param_q_scalar_q.exp()
        stat = quantizer(
            z_pos=z_res,
            var_q_pos=param_q,
            codebook=codebook,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det
        )

        # Residual/rurrent latent
        z_q = stat["z_q"]
        z_res = z_res - z_q
        z_cur = z_cur + z_q

        return z_cur, z_res, stat
    
    def decode_from_latent(self, x, z_quantized):
        # z_quantized: z_q
        # x : z_res
        diff_z = x - z_quantized 
        return z_quantized, diff_z

    def get_indices(self, activation, quantizer, codebook, log_param_q_scalar_q):
        param_q = log_param_q_scalar_q.exp()
        return quantizer.get_index(z_pos=activation, var_q_pos=param_q, codebook=codebook)


    def get_quantized_vector(self, indices, quantizer, codebook, bs, dim_z, width, height):
        return quantizer.get_vector(indices=indices, codebook=codebook, bs=bs, dim_z=dim_z, width=width, height=height)
    

class TopDown(nn.Module):
    def __init__(self, num_residual, flg_codebook_share=False):
        super().__init__()
        self.num_residual = num_residual
        self.flg_codebook_share = flg_codebook_share
        
        # Quantizer
        blocks_sq = []
        for i in range(num_residual):
            blocks_sq.append(TopDownBlock())
        self.blocks_sq = nn.ModuleList(blocks_sq)
        
    def forward(self, activation, quantizer, codebook, log_param_q_scalar_q, flg_train, flg_quant_det):
        stats = []
        for idx, block in enumerate(self.blocks_sq):

            # Initialization
            if idx == 0:
                z_res = activation
                z_cur = torch.zeros_like(z_res)

            # Codebook is shared or not
            if self.flg_codebook_share:
                idx_codebook = 0
            else:
                idx_codebook = idx

            # Quantization block
            z_cur, z_res, block_stats = block(
                z_cur=z_cur,
                z_res=z_res,
                quantizer=quantizer[idx],
                codebook=codebook[idx_codebook],
                log_param_q_scalar_q=log_param_q_scalar_q[:idx+1],
                flg_train=flg_train,
                flg_quant_det=flg_quant_det
            )
            stats.append(block_stats)
            
        return z_cur, stats
    

    def forward_uncond(self, x, quantizer, codebook, log_param_q_scalar, flg_train, flg_quant_det):
        for idx, block in enumerate(self.blocks_sq):
            x = block.forward_uncond(x, quantizer[idx], codebook[idx], log_param_q_scalar[idx], flg_train, flg_quant_det)
        return x
    
    def decode_from_latents(self, z_quantized_list):
        # z_quantized_list: codebook_num
        for idx, block in enumerate(self.blocks_sq):
            if idx == 0:
                z_all, diff_z = block.decode_from_latent(0, z_quantized_list[idx])
                z_gen = z_all
            else:
                z, diff_z = block.decode_from_latent(diff_z, z_quantized_list[idx])
                z_gen = z_gen + z
        return z_gen
    
    def get_indices_list(self, activation, quantizer, codebook, log_param_q_scalar_q):
        indices_list = []
        for idx, blk in enumerate(self.blocks_sq):
            indices, z_q = blk.get_indices(activation, quantizer[idx], codebook[idx], log_param_q_scalar_q)
            activation = activation - z_q
            indices_list.append(indices)
        return indices_list
    
    def get_quantized_vector_list(self, indices_list, quantizer, codebook, bs, dim_z, width, height):
        z_quantized_list = []
        for idx, blk in enumerate(self.blocks_sq):
            z_quantized = blk.get_quantized_vector(indices_list[idx], quantizer[idx], codebook[idx], bs, dim_z, width, height)
            z_quantized_list.append(z_quantized)
        return z_quantized_list
    
    
    