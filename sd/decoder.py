#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from attention import SelfAttention
from torch import nn
from torch.nn import functional as F


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.group_norm(x)
        x = F.silu(x)
        B, C, H, W = x.size()
        # B, C, H, W -> B, C, H * W
        x = x.view(b, c, h * w)
        # B, C, H * W -> B, H * W, C
        x = x.transpose(-1, -2)
        # B, H * W, C -> B, H * W, C
        x = self.attention(x)
        # B, H * W, C -> B, C, H * W
        x = x.transpose(-1, -2)
        # B, C, H * W -> B, C, H, W
        x = x.view(B, C, H, W)
        x = x + residue
        return x


class VAE_ResidualBlock(nn.module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def fowward(self, x: torch.Tensor) -> torch.Tensor:
        # B, in_channels, H, W
        residue = x
        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residue)
        return x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # B, 512, H/8, W/8 -> B, 512, H/8, W/8
            VAE_ResidualBlock(512, 512),
            # B, 512, H/8, W/8 -> B, 512, H/4, W/4
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # B, 512, H/4, W/4 -> B, 512, H/2, W/2
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # B, 256, H/2, W/2 -> B, 256, H, W
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(x: torch.Tensor) -> torch.Tensor:
        # x: B, 4, H/8, W/8
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x
