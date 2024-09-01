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

import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.nheads = n_heads
        self.d_head = d_embed // n_heads

    def forward(x: torch.Tensor, causal_mask=False):
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape
        assert embed_dim % self.nheads == 0
        intermediate_shape = (batch_size, seq_len, self.nheads, self.d_head)
        q, k, v = self, in_proj(x).chunk(3, dim=-1)

        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-2, -1)
        if causal_mask:
            caussal_mask = torch.ones(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(causal_mask, float("-inf"))

        weight = F.softmax(weight / math.sqrt(self.d_head), dim=-1)
        # B, N, S, S @ B, N, S, H -> B, N, S, H
        output = weight @ v
        output = output.transpose(1, 2).reshape(input_shape)
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()
        self.in_proj_q = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.in_proj_k = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.in_proj_v = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.nheads = n_heads
        self.d_head = d_embed // n_heads

    def forward(x: torch.Tensor, context: torch.Tensor):
        # x: batch_size, seq_len_q, embed_dim_q
        # context: batch_size, seq_len_kv, embed_dim_kv -> batch_size, 77, 768
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape
        assert embed_dim % self.nheads == 0

        interim_shape = (batch_size, -1, self.nheads, self.d_head)

        q = self.in_proj_q(x).view(interim_shape).transpose(1, 2)
        k = self.in_proj_k(context).view(interim_shape).transpose(1, 2)
        v = self.in_proj_v(context).view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-2, -1)
        weight = F.softmax(weight / math.sqrt(self.d_head), dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous().view(input_shape)
        output = self.out_proj(output)
        return output
