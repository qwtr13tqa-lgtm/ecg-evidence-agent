# modules2.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerEncoderBlock(nn.Module):
    "A single Transformer encoder block with self-attention and FFN."

    def __init__(self, d_model, h, d_ff, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer_norm1 = LayerNorm(d_model)
        self.sublayer_norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sub-layer 1: Self-attention
        x = self.sublayer_norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        # Sub-layer 2: Position-wise Feed-Forward
        x = self.sublayer_norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class Encoder1D(nn.Module):
    # Added output_dim parameter
    def __init__(self, nc, output_dim=50):
        super(Encoder1D, self).__init__()
        ndf = 32
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Output channel now uses output_dim
            nn.Conv1d(ndf * 16, output_dim, 15, 1, 0),
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Decoder1D(nn.Module):
    # Added input_dim parameter
    def __init__(self, nc, input_dim=50):
        super(Decoder1D, self).__init__()
        ngf = 32
        self.main = nn.Sequential(
            # Input channel now uses input_dim
            nn.ConvTranspose1d(input_dim, ngf * 16, 15, 1, 0),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Encoder2D(nn.Module):
    # Added output_dim parameter
    def __init__(self, nc, output_dim=50):
        super(Encoder2D, self).__init__()
        ndf = 32
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 1, 0),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Output channel now uses output_dim
            nn.Conv2d(ndf * 16, output_dim, 3, 1, 0),
        )

    def forward(self, input):
        output = self.main(input)
        return output


# --- NEW: Memory Module for Anomaly Detection ---
class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input shape: (B, N, C)
        s = input.data.shape
        l = len(s)

        # reshape input to (B*N, C)
        x = input.contiguous().view(-1, s[-1])

        # Calculate Cosine Similarity
        att_weight = F.linear(x, self.weight)  # Fea x Mem^T
        att_weight = F.softmax(att_weight, dim=1)  # Softmax normalization

        # Hard Shrinkage (Optional: makes memory sparse)
        if self.shrink_thres > 0:
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)  # Re-normalize

        output = F.linear(att_weight, self.weight.t())  # Att * Mem

        # Reshape back to (B, N, C)
        output = output.view(s)
        return output, att_weight

    def hard_shrink_relu(self, input, lambd=0, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output