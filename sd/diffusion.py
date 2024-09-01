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
from torch import nn
from torch.nn import functional as F

from attention import CrossAttention, SelfAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(320, embedding_dim * 4)
        self.linear_2 = nn.Linear(embedding_dim * 4, embedding_dim * 4)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(time)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, UNET_AttentionBlock):
                x = module(x, context)
            elif isinstance(module, UNET_ResidualBlock):
                x = module(x, time)
            else:
                x = module(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W -> B, C, H*2, W*2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, output_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, 320, H/8, W/8
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv(x)
        # x: B, 4, H/8, W/8
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280):
        super().__init__()
        self.grpup_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: B, in_channels, height. weight
        # time: 1, 1280
        residual = x
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = x + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.group_norm_2(merged)
        merged = F.silu(merged)
        merged = self.conv_2(merged)
        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_context: int = 768):
        super().__init__()
        channels = n_embed * n_heads
        self.group_norm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads, channels, d_context, in_proj_bias=False
        )
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: B, in_channels, height. weight
        # context: B, Seq_len, Embedding_dim
        residual_long = x
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv_input(x)
        b, c, h, w = x.shape
        # B, channels, height, weight -> B, channels, height * weight
        x = x.view(b, c, h * w)
        # B, channels, height * weight -> B, height * weight, channels
        x = x.transpose(-1, -2)
        # normalization + self-attention with skip connection
        residual_short = x
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x = x + residual_short

        # normalization + cross-attention with skip connection
        residual_short = x
        x = self.layer_norm_2(x)
        x = self.attention_2(x, context)
        x = x + residual_short

        # feedforward with skip connection
        residual_short = x
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + residual_short

        x = x.transpose(-1, -2)
        x = x.view(b, c, h, w)
        x = self.conv_output(x) + residual_long
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                # B, 4, H/8, W/8 -> B, 320, H/8, W/8
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(UNET_ResidualBlock(320, 320)),
                UNET_AttentionBlock(8, 40),
                SwitchSequential(UNET_ResidualBlock(320, 320)),
                UNET_AttentionBlock(8, 40),
                # B, 320, H/8, W/8 -> B, 320, H/16, W/16
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(UNET_ResidualBlock(320, 640)),
                UNET_AttentionBlock(8, 80),
                SwitchSequential(UNET_ResidualBlock(640, 640)),
                UNET_AttentionBlock(8, 80),
                # B, 640, H/16, W/16 -> B, 640, H/32, W/32
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(UNET_ResidualBlock(640, 1280)),
                UNET_AttentionBlock(8, 160),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                UNET_AttentionBlock(8, 160),
                # B, 1280, H/32, W/32 -> B, 1280, H/64, W/64
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
            ]
        )

        def forward(self, x, context, time):
            # x: (Batch_Size, 4, Height / 8, Width / 8)
            # context: (Batch_Size, Seq_Len, Dim)
            # time: (1, 1280)

            skip_connections = []
            for layers in self.encoders:
                x = layers(x, context, time)
                skip_connections.append(x)

            x = self.bottleneck(x, context, time)

            for layers in self.decoders:
                # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
                x = torch.cat((x, skip_connections.pop()), dim=1)
                x = layers(x, context, time)

            return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(
        self, lantent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent: B, 4, H/8, W/8
        # context: B, Seq_len, Embedding_dim
        # time: 1, 320
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # B, 4, H/8, W/8 -> B, 320, H/8, W/8
        output = self.unet(lantent, context, time)
        # B, 320, H/8, W/8 -> B, 4, H, W
        output = self.final(output)
        return output
