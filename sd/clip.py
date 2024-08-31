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

from attention import SelfAttention


class ClipEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        x = self.token_embedding(tokens)
        x = x + self.position_embedding
        return x


class ClipLayer(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int):
        super().__init__()
        self.attention = SelfAttention(num_heads, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.norm1(x)
        x = self.attention(x, causal_mask=True)
        x = x + residue

        residue = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGelu activation function
        x = self.linear2(x)
        x = x + residue
        return x


class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbedding(49408, 768, 77)
        self.layers = nn.Module([ClipLayer(12, 768) for i in range(12)])
        self.norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.LongTensor)
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
