import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def sample_gumbel(shape, device, eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size(), logits.device)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances



class VectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        return self._quantize(z_from_encoder, param_q, codebook,
                                flg_train=flg_train, flg_quant_det=flg_quant_det)
    
    def _quantize(self):
        raise NotImplementedError()
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_codes(self):
        raise NotImplementedError()
    
    def _calc_distance_bw_enc_dec(self):
        raise NotImplementedError()


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, flg_loss_continuous, temperature=0.5, param_var_q_prior=None, param_var_q_posterior="gaussian_1", device="cuda"):
        super(GaussianVectorQuantizer, self).__init__()
        # parameter details.
        self.device = device
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
        self.param_var_q_prior = param_var_q_prior
        self.param_var_q_posterior = param_var_q_posterior
        self.flg_loss_continuous = flg_loss_continuous
    
    def forward(self, z_pos, var_q_pos, codebook, flg_train, flg_quant_det=False, z_pri=None, var_q_pri=None):
        return self._quantize(z_pri, z_pos, var_q_pri, var_q_pos, codebook,
                                flg_train=flg_train, flg_quant_det=flg_quant_det)    
    
    # TODO: 
    def get_index(self, z_pos, var_q_pos, codebook):
        # z_pos: B, C, H, W
        bs, dim_z, width, height = z_pos.shape
        z_pos_permuted = z_pos.permute(0, 2, 3, 1).contiguous()     # B, H, W, C
        if torch.numel(var_q_pos) > 1:
            var_q_pos_main = var_q_pos[-1]
        else:
            var_q_pos_main = var_q_pos
        precision_q_pos = 1. / torch.clamp(var_q_pos_main, min=1e-10)
        logit_pos = -self._calc_distance_bw_enc_codes(z_pos_permuted, codebook, 0.5 * precision_q_pos, self.param_var_q_posterior)
        # prob_pos = torch.softmax(logit_pos, dim=-1)
        # log_prob_pos = torch.log_softmax(logit_pos, dim=-1)
        indices = torch.argmax(logit_pos, dim=1).unsqueeze(1)
        encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=self.device)
        encodings_hard.scatter_(1, indices, 1)
        z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        return indices, z_to_decoder
        
    # TODO:  
    def get_vector(self, indices, codebook, bs, dim_z, width, height):
        encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=self.device)
        encodings_hard.scatter_(1, indices, 1)
        z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        return z_to_decoder


    def _quantize(self, z_pri, z_pos, var_q_pri, var_q_pos, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_pos.shape

        if self.param_var_q_prior is not None:
            z_pri_permuted = z_pri.permute(0, 2, 3, 1).contiguous()
            precision_q_pri = 1. / torch.clamp(var_q_pri, min=1e-10)
            logit_pri = -self._calc_distance_bw_enc_codes(z_pri_permuted, codebook, 0.5 * precision_q_pri, self.param_var_q_prior)
            log_prob_pri = torch.log_softmax(logit_pri, dim=-1)
        else:
            log_prob_pri = 0.0
        
        z_pos_permuted = z_pos.permute(0, 2, 3, 1).contiguous()
        if torch.numel(var_q_pos) > 1:
            var_q_pos_main = var_q_pos[-1]
        else:
            var_q_pos_main = var_q_pos
        precision_q_pos = 1. / torch.clamp(var_q_pos_main, min=1e-10)
        logit_pos = -self._calc_distance_bw_enc_codes(z_pos_permuted, codebook, 0.5 * precision_q_pos, self.param_var_q_posterior)
        prob_pos = torch.softmax(logit_pos, dim=-1)
        log_prob_pos = torch.log_softmax(logit_pos, dim=-1)
        
        # Quantization
        if flg_train:
            indices = torch.argmax(logit_pos, dim=1)
            encodings = gumbel_softmax_sample(logit_pos, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(prob_pos.detach(), dim=0)
        else:
            if flg_quant_det:
                #---------------------------------- 改变索引 ------------------------
             
                #-------------------------------------------------------------------
                indices = torch.argmax(logit_pos, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=self.device)
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(prob_pos)
                indices = dist.sample()
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(prob_pos, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(prob_pos * (log_prob_pos - log_prob_pri), dim=(0,1)) / bs
        if self.flg_loss_continuous:
            precision_q_pos_sum = 1. / torch.clamp(var_q_pos.sum(), min=1e-10)
            kld_continuous = self._calc_distance_bw_enc_dec(z_pos, z_to_decoder, 0.5 * precision_q_pos_sum).mean()
            loss = kld_discrete + kld_continuous
        else:
            loss = kld_discrete
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return dict(kl=loss, z_q=z_to_decoder, perplexity=perplexity, avg_probs=avg_probs, indices=indices.unsqueeze(1).view(bs, width, height).contiguous())
    
    def set_temperature(self, value):
        self.temperature = value

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight, param_var_q=None):       
        if param_var_q == "gaussian_1":     # z_from_encoder: B, H, W, C; codebook: K, C; self.dim_dict: C
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif param_var_q == "gaussian_2":
            weight = weight.tile(1, 1, 8, 8).view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif param_var_q == "gaussian_3":
            weight = weight.view(-1,1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif param_var_q == "gaussian_4":
            z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict).unsqueeze(2)
            codebook = codebook.t().unsqueeze(0)
            weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_dict).unsqueeze(2)
            distances = torch.sum(weight * ((z_from_encoder_flat - codebook) ** 2), dim=1)

        return distances
        
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2,3))
    
