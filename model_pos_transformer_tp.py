import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
import os
from timm import create_model

from tensor2img import *
from visual_patch import *
from cut_simple import *
from tensor2img import *
from visual_patch import *
from patch_simple_torch2 import *
from gauss import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math
import torch
import torch.nn as nn
import numpy as np


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim))
        self.temporal_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B_t, t, N, D = x.shape
        if t > self.temporal_pos_embed.shape[1]:
            raise ValueError(f"err:{self.temporal_pos_embed.shape[1]}")
        x = x + self.temporal_pos_embed[:, :t, :].unsqueeze(2)  # (B_t, t, N, D)

        x = x.view(B_t * N, t, D)

        for blk in self.temporal_blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.view(B_t, t, N, D)  # (B_t, t, N, D)
        return x


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, embed_dim, patches, max_theta=math.pi, max_phi=2 * math.pi):
        super().__init__()
        self.embed_dim = embed_dim
        self.patches = patches
        self.max_theta = max_theta
        self.max_phi = max_phi

        # ʹ10000^(2i/d_model)
        div_term_denominator = torch.exp((2 * torch.arange(0, embed_dim // 4)) * math.log(10000.0) / embed_dim)
        self.register_buffer('div_term_denominator', div_term_denominator)

        pos_encodings = []
        for patch in patches:
            theta_center = (patch['theta_min'] + patch['theta_max']) / 2
            phi_center = (patch['phi_min'] + patch['phi_max']) / 2
            pos_encoding = self.get_position_encoding(theta_center, phi_center)
            pos_encodings.append(pos_encoding)

        self.pos_embed = nn.Parameter(torch.stack(pos_encodings).unsqueeze(0))
        self.pos_embed.requires_grad = True

    def get_position_encoding(self, theta, phi):
        phi = phi % (2 * math.pi)
        theta_norm = theta / self.max_theta
        phi_norm = phi / self.max_phi

        position_encoding = torch.zeros(self.embed_dim)

        for i in range(0, self.embed_dim, 4):
            div_term = self.div_term_denominator[i // 4]
            position_encoding[i] = math.sin(theta_norm / div_term)
            position_encoding[i + 1] = math.cos(theta_norm / div_term)
            position_encoding[i + 2] = math.sin(phi_norm / div_term)
            position_encoding[i + 3] = math.cos(phi_norm / div_term)

        return position_encoding

    def forward(self):
        return self.pos_embed


def spherical_positional_encoding(D, patches):

    num_patches = len(patches)
    pos_enc = torch.zeros(num_patches, D)

    if D % 4 != 0:
        raise ValueError("Dimension D must be divisible by 4 for spherical encoding")

    for i, patch in enumerate(patches):

        center_theta = (patch['theta_min'] + patch['theta_max']) / 2
        center_phi = (patch['phi_min'] + patch['phi_max']) / 2

        for j in range(0, D // 2, 2):

            freq = 10000 ** (2 * j / D)
            pos_enc[i, j] = math.sin(center_theta / freq)
            pos_enc[i, j + 1] = math.cos(center_theta / freq)
            pos_enc[i, D // 2 + j] = math.sin(center_phi / freq)
            pos_enc[i, D // 2 + j + 1] = math.cos(center_phi / freq)

    return pos_enc


class SphericalPositionalEncoding(nn.Module):

    def __init__(self, D, patches):
        super().__init__()
        if D % 4 != 0:
            raise ValueError("Dimension D must be divisible by 4 for spherical encoding")
        self.D = D
        self.num_patches = len(patches)
        pos_enc = self._compute_encoding(patches)
        self.register_buffer('pos_enc', pos_enc)
    def _compute_encoding(self, patches):
        pos_enc = torch.zeros(self.num_patches, self.D)
        freqs = 10000 ** (2 * torch.arange(self.D // 4) / self.D)

        for i, patch in enumerate(patches):
            center_theta = (patch['theta_min'] + patch['theta_max']) / 2.0
            center_phi = (patch['phi_min'] + patch['phi_max']) / 2.0
            for j in range(self.D // 4):
                freq = freqs[j]
                pos_enc[i, 2 * j] = math.sin(center_theta / freq)
                pos_enc[i, 2 * j + 1] = math.cos(center_theta / freq)
                pos_enc[i, self.D // 2 + 2 * j] = math.sin(center_phi / freq)
                pos_enc[i, self.D // 2 + 2 * j + 1] = math.cos(center_phi / freq)
        return pos_enc

    def forward(self):
        return self.pos_enc





class RobustDynamicViT(nn.Module):
    def __init__(self, original_vit, patches):
        super().__init__()

        # self.pos_embed = nn.Parameter(original_vit.pos_embed.data.clone())
        # self.pos_embed.requires_grad = True
        self.embed_dim = original_vit.embed_dim
        # self.absolute_pos_encoding = AbsolutePositionEncoding(self.embed_dim, patches)
        self.absolute_pos_encoding = SphericalPositionalEncoding(self.embed_dim, patches)
        # Patch Embedding
        self.patch_embed = nn.Linear(3 * 16 * 16, self.embed_dim)
        # self.patch_embed = nn.Conv2d(3 * 16 * 16, self.embed_dim, kernel_size=1, stride=1)
        conv_weight = original_vit.patch_embed.proj.weight
        self.patch_embed.weight.data = conv_weight.view(self.embed_dim, -1)
        self.patch_embed.bias.data = original_vit.patch_embed.proj.bias
        self.blocks = original_vit.blocks
        self.norm = original_vit.norm
        self.conv3 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1)  # (kernel_size - stride) // 2
        )
        self.temporal_transformer = TemporalTransformer(
            embed_dim=self.embed_dim,
            num_heads=8,
            num_layers=4
        )
        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1 = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )


        pre_model_path = '/weights/model_pos_svgc/vr/vr_final.pth'
        if os.path.exists(pre_model_path):
            # self.load_state_dict(torch.load(pre_model_path,map_location=device).state_dict())
            self.load_state_dict(torch.load(pre_model_path, map_location=device).state_dict(), strict=False)
            print("Load pre-trained weights")


    def forward(self, x, meta_list, shape_r_out, shape_c_out, t, cb):
        B, N, C, H, W = x.shape #(10,110,3,16,16)
        assert H == 16 and W == 16, "err"
        x_embed = self.patch_embed(x.view(B * N, -1)).view(B, N, -1) #(10,110,768)

        pos_embed = self.absolute_pos_encoding()
        # pos_embed = self.absolute_pos_encoding

        x_embed += pos_embed
        for blk in self.blocks:
            x_embed = blk(x_embed)
        x_embed = self.norm(x_embed)
        # y = x_embed.view(B, N, 3, H, W) #B=B'*T, N=num_patches


        B_t = int(B / t)
        y = x_embed.view(B_t, t, N, -1)  # (B_t, t, N, D)
        y = self.temporal_transformer(y)  # (B_t, t, N, D)
        y = y.view(B, N, 3, H, W)  #(B, N, 3, 16, 16)


        y = reconstruct_from_patches(y, meta_list, (shape_r_out, shape_c_out)).float().to(device)
        y = self.conv3(y)
        z=y

        #prior
        y = self.conv0(y)
        if cb.size(0) != y.size(0):
            cb = cb[:y.size(0)]
        y = torch.cat([y, cb], dim=1)
        y = self.conv1(y)+z

        y = torch.sigmoid(y)
        y = gaussmap(y) / 255

        return y

