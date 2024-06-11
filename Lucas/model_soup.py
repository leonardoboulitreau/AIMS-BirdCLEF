from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim


'''
MetaStyleSpeech style encoder https://arxiv.org/pdf/2106.03153.pdf

Adapted from: https://github.com/keonlee9420/StyleSpeech

'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FCBlock(nn.Module):
    """ Fully Connected Block """

    def __init__(self, in_features, out_features, activation=None, bias=False, dropout=None, spectral_norm=False):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1DBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None, dropout=None, spectral_norm=False):
        super(Conv1DBlock, self).__init__()

        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module(
            "conv_layer",
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                spectral_norm=spectral_norm,
            ),
        )
        if activation is not None:
            self.conv_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = self.conv_layer(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)

        x = x.contiguous().transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        return x


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, layer_norm=False, spectral_norm=False):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_ks = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_vs = LinearNorm(d_model, n_head * d_v, spectral_norm=spectral_norm)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.fc = LinearNorm(n_head * d_v, d_model, spectral_norm=spectral_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output
    
def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    	
    #print(f'lengths = {lengths}')
    #print(f'max_len = {max_len}')
    
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool() # (B, max_len)
    return mask


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class XCAMetaBird(nn.Module):
    """ Mel-Style Encoder """

    def __init__(self, num_mel, embedding_dim, num_classes, n_spectral = 2, n_temporal = 2, n_mha = 2, mha_heads = 1, max_seq_len = 400):
        super(XCAMetaBird, self).__init__()
        self.num_classes = num_classes
        # n_position = model_config["max_seq_len"] + 1
        n_mel_channels = num_mel
        d_melencoder = embedding_dim
        n_spectral_layer = n_spectral
        n_temporal_layer = n_temporal
        n_slf_attn_layer = n_mha
        n_slf_attn_head = mha_heads
        d_k = d_v = (
            embedding_dim
            // mha_heads
        )
        kernel_size = 3
        dropout = 0.2

        self.max_seq_len = max_seq_len # This is our default, but it is interesting to be adaptive

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder)

        self.spectral_stack = nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        
        self.temporal_stack = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        d_melencoder, 2 * d_melencoder, kernel_size, activation=Mish(), dropout=dropout
                    ),
                    # nn.AvgPool1d(3, stride=1, padding=1),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = nn.ModuleList(
            [
                XCA(
                    d_melencoder, n_slf_attn_head
                )
                for _ in range(n_slf_attn_layer)
            ]
        )
        

        self.fc_2 = FCBlock(d_melencoder, d_melencoder)

        self.classifier_layer = nn.Linear(embedding_dim, self.num_classes)
        self.attn = nn.Linear(embedding_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, mel):
        # mel -> (BS, MEL_DIM, L)
        # print(f'mel shape = {mel.shape}')

        # Tranposing mel input to be (L, MEL_DIM)
        mel = mel.permute(0,2,1) # -> (BS, L, MEL_DIM)

        # Pre-net per frame
        enc_output = self.fc_1(mel) # -> (BS, L, EMB_SIZE)
        # print(f'enc_output shape = {enc_output.shape}')

        # Spectral Processing per frame
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output) # -> (BS, L, EMB_SIZE)
        # print(f'after spectral process shape = {enc_output.shape}')


        # Temporal Processing (only per frame to get temporal patterns)
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output # -> (BS, L, EMB_SIZE)

        # print(f'after temporal process shape = {enc_output.shape}')
        
        # Multi-head self-attention per frame
        for _, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output = layer(
                enc_output
            )
            enc_output = residual + enc_output # -> (BS, L, EMB_SIZE)
        # print(f'after mha process shape = {enc_output.shape}')
        
        # Post-net for frames and frequency
        enc_output = self.fc_2(self.activation(enc_output)) # -> (BS, L, EMB_SIZE)


        attn_weights = F.softmax(self.attn(enc_output), dim=1) # -> (BS, 2, 1)
        # print(attn_weights.shape)
        enc_output = torch.sum(enc_output * attn_weights, dim=1) # -> (BS, EMB_SIZE)
        
        pred = self.classifier_layer(self.dropout(enc_output)) # -> (BS, NUM_CLASSES)

        return pred