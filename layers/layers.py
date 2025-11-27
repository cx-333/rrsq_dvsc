# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2

# same to subpel_conv3x3
class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, to_cat, cat_at_front)

    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )


    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        return self.forward_torch(x, quant_step, to_cat, cat_at_front)


    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out


#----------------------------------------------------------------------------------------------------
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class GatedFFN(nn.Module):
    # effective than AdditiveFFN
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.fc2 = nn.Conv2d(in_features, hidden_features, 1)

        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out1 = self.act(self.fc1(x))
        out2 = self.fc2(x)
        out = out1 * out2
        out =self.drop(out)
        out = self.fc3(out)
        out - self.drop(out)
        return out
    
    

class CrossWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, kv, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = kv.shape
        qB_, _, _ = x.shape

        kv = self.kv(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q = self.q(x).reshape(qB_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(
                0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
    
class S2CAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window_attn = WindowAttention(dim, (window_size, window_size), num_heads, mlp_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GatedFFN(dim, mlp_hidden_dim, dim, act_layer=nn.Sigmoid)

        self.window_size = window_size

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0
        x = self.norm1(x.permute(0, 2, 3, 1)).contiguous().permute(0, 3, 1, 2)
        x = window_partition(x, self.window_size)       # B*num_windows, window_size, window_size, C
        x = x.view(-1, self.window_size*self.window_size, C) + self.window_attn(x.view(-1, self.window_size*self.window_size, C))
        # B * num_windows, window_size*window_size, C
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self.norm2(window_reverse(x, self.window_size, H, W)).contiguous().permute(0, 3, 1, 2)
        x = x + self.mlp(x)
        return x
    
class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # mlp implemented by convolution neural layer: Vanilla-FFN
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class SwinDecoder(nn.Module):
    def __init__(self, dim, num_heads, shift_size=0, window_size=8, mlp_ratio=4., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop_path=0.1, drop=0.1):
        super().__init__()
        # query -> self-atteion, query,key,value -> cross-attention
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.cross_attn = CrossWindowAttention(
            dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_attn_mask(self, h, w, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = h, w
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        attn_mask = attn_mask if attn_mask is None else attn_mask.to(device)
        # self.register_buffer("attn_mask", attn_mask)
        self.attn_mask = attn_mask

    def forward(self, x, kv):
        # x, kv: B, C, H, W; x->query; kv->key,value
        B, C, H, W = x.shape

        self.create_attn_mask(H, W, x.device)

        shortcut = x
        # x = x.view(B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        kv = kv.permute(0, 2, 3, 1)

        # cyclic shift
        if self.shift_size > 0:
            # shifted_x = x
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_kv = kv

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        kv_windows = window_partition(shifted_kv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        kv_windows = kv_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA; self-attention
        # nW*B, window_size*window_size, C
        attn_windows = self.norm1(x_windows + self.attn(x_windows, mask=self.attn_mask))

        # W-MSA/SW-MSA cross-attention
        attn_windows = self.norm1(attn_windows + self.cross_attn(attn_windows, kv_windows, mask=self.attn_mask))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x)).permute(0, 2, 1).view(B, C, H, W)

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x).permute(0, 2, 3, 1).view(B, H*W, C))).permute(0, 2, 1).view(B, C, H, W)
        # B, C, H, W
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, H, W):
        flops = 0
        # norm1
        flops += (self.dim * H * W) * 2
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size) * 2
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    # (B, C, H, W) -> (B, 2*C, H//2, W//2)
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(2*dim)

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        x = x.permute(0, 2, 3, 1).contiguous()      # B, H, W, C

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)        # B, H//2*W//2, 2*C
        x = x.permute(0, 2, 1).view(B, 2*C, H//2, W//2)     # B, 2*C, H//2, W//2
        return x

    def flops(self, H, W):
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchReverseMerging(nn.Module):
    # (B, C, H, W) -> (B, C, 2H, 2W)
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchReverseMerging, self).__init__()
        self.dim = dim
        self.increment = nn.Linear(dim, dim*4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x = self.norm(x)
        x = self.increment(x)       # B, HW, C*4
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, 4*C, H, W
        x = nn.PixelShuffle(2)(x)
        return x

    def flops(self, H, W):
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops

    

class TSEBasicBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, downsample=2, window_size=8, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 drop_path=0., drop=0.):
        super().__init__()  
        self.dim = dim      # channel dim
        self.blocks = nn.ModuleList([
            SwinDecoder(
                dim=dim, num_heads=num_heads,
                shift_size= 0 if (i % 2 == 0) else window_size // 2, 
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                drop=drop
            ) for i in range(depth)
        ])
        
        if downsample:
            self.downsample = nn.Conv2d(dim, dim*downsample, kernel_size=3, stride=2, padding=1)
            self.kv_downsample = nn.Conv2d(dim, dim*downsample, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x, kv):
        for blk in self.blocks:
            x = blk(x, kv)
        if self.downsample is not None:
            x = self.downsample(x)
            kv = self.kv_downsample(kv)
        return x, kv
          

class TSRBasicBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, upsample=2, window_size=8, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 drop_path=0., drop=0.):
        super().__init__()  
        self.dim = dim      # channel dim
        self.blocks = nn.ModuleList([
            SwinDecoder(
                dim=dim, num_heads=num_heads,
                shift_size= 0 if (i % 2 == 0) else window_size // 2, 
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                drop=drop
            ) for i in range(depth)
        ])
        
        if upsample:
            self.upsample = SubpelConv2x(dim, dim//upsample, 3, padding=1)
            self.kv_upsample = SubpelConv2x(dim, dim//upsample, 3, padding=1)
        else:
            self.upsample = None

    def forward(self, x, kv):
        for blk in self.blocks:
            x = blk(x, kv)
        if self.upsample is not None:
            x = self.upsample(x)
            kv = self.kv_upsample(kv)
        return x, kv

# TSE; temporal spatial extractor
class TSE(nn.Module):
    def __init__(self, 
                 dim=[128, 256], 
                 depth=[2, 4], 
                 num_heads=[8, 8], 
                 downsample=[True, True], 
                 window_size=8, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 drop_path=0., drop=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(dim[0], dim[0], 1) # query
        self.conv2 = nn.Conv2d(dim[0], dim[0], 1) # key, value     
        
        self.layers = nn.ModuleList([
            TSEBasicBlock(
                dim=dim[i], 
                depth=depth[i],
                num_heads=num_heads[i],
                downsample=downsample[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                drop=drop,
            ) for i in range(len(depth))
        ])
     
    def forward(self, x, kv):
        x = self.conv1(x)
        kv = self.conv2(kv)
        for blk in self.layers:
            x, kv = blk(x, kv)
        return x
       

# TSR; temporal spatial reconstrcution
class TSR(nn.Module):
    def __init__(self, 
                 dim=[256, 128], 
                 depth=[4, 2], 
                 num_heads=[8, 8], 
                 upsample=[True, True], 
                 window_size=8, 
                 mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 drop_path=0., drop=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(dim[0], dim[0], 1) # query
        self.conv2 = nn.Conv2d(dim[0], dim[0], 1) # key, value
        
        self.layers = nn.ModuleList([
            TSRBasicBlock(
                dim=dim[i], 
                depth=depth[i],
                num_heads=num_heads[i],
                upsample=upsample[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                drop=drop,
            ) for i in range(len(depth))
        ])
        
    def forward(self, x, kv):
        x = self.conv1(x)
        kv = self.conv2(kv)
        for blk in self.layers:
            x, kv = blk(x, kv)
        return x
    
class CrossWindowFusion(nn.Module):
    def __init__(self, dim, num_heads, shift_size=0, window_size=8, mlp_ratio=4., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop_path=0., drop=0.):
        super().__init__()
        # input: q, kv (B, C, H, W)  -> output: (B, C, H, W)
        # query -> self-atteion, query,key,value -> cross-attention
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.cross_attn = CrossWindowAttention(
            dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # attn_mask = None
        # self.register_buffer("attn_mask", attn_mask)
        # self.attn_mask = nn.Parameter(None)

    def create_attn_mask(self, h, w, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = h, w
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        attn_mask = attn_mask if attn_mask is None else attn_mask.to(device)
        # self.register_buffer("attn_mask", attn_mask)
        self.attn_mask = attn_mask

    def forward(self, x, kv):
        # x, kv: B, C, H, W; x->query; kv->key,value
        B, C, H, W = x.shape

        self.create_attn_mask(H, W, x.device)

        shortcut = x
        # x = x.view(B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        kv = kv.permute(0, 2, 3, 1)

        # cyclic shift
        if self.shift_size > 0:
            # shifted_x = x
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_kv = kv

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        kv_windows = window_partition(shifted_kv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        kv_windows = kv_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA; self-attention
        # nW*B, window_size*window_size, C
        attn_windows = self.norm1(x_windows + self.attn(x_windows, mask=self.attn_mask))

        # W-MSA/SW-MSA cross-attention
        attn_windows = self.norm1(attn_windows + self.cross_attn(attn_windows, kv_windows, mask=self.attn_mask))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x)).permute(0, 2, 1).view(B, C, H, W)

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x).permute(0, 2, 3, 1).view(B, H* W, C))).permute(0, 2, 1).view(B,
                                                                                                                    C,
                                                                                                                    H,
                                                                                                                    W)
        # B, C, H, W
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, H, W):
        flops = 0
        # norm1
        flops += (self.dim * H * W) * 2
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size) * 2
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    