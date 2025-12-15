import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import logging

class PositionalEncoder(nn.Module):
    """Standard Absolute Positional Encoding"""
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._max_seq_length = max_seq_length
        
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)

        @torch.jit.script
        def splice_by_size(source, target):
            length = target.size(1);
            return source[:, :length]
        self.splice_by_size = splice_by_size
    
    def forward(self, x):
        if(x.shape[1] > self._max_seq_length):
            # logging.warn("Input longer than max_seq_length")
            x = x[:, :self._max_seq_length]
        
        x = x * math.sqrt(self.d_model)
        spliced_pe = self.splice_by_size(self.pe, x)
        pe = spliced_pe.requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention"""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        value, attn = self.attention(q, k, v, mask, self.dropout)
        concat = value.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, attn

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        scores = functional.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

# ============================================================================
# THÊM LỚP MỚI: RelativeMultiHeadAttention (Cho Re-Transformer)
# ============================================================================
class RelativeMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Relative Positional Encoding (Shaw et al., 2018)"""
    def __init__(self, heads, d_model, dropout=0.1, max_relative_position=16):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.max_relative_position = max_relative_position

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # Relative Positional Embeddings (cho Key và Value - hoặc chỉ Key tùy cài đặt)
        # Ở đây cài đặt đơn giản theo Shaw et al.: chỉ cộng vào Key
        vocab_size_pe = max_relative_position * 2 + 1
        self.relative_pe_k = nn.Embedding(vocab_size_pe, self.d_k)
        # self.relative_pe_v = nn.Embedding(vocab_size_pe, self.d_k) # Tùy chọn

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        len_q = q.shape[1]
        len_k = k.shape[1] # len_k = len_v

        # 1. Linear Projections
        q_p = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2) # [bs, h, len_q, d_k]
        k_p = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2) # [bs, h, len_k, d_k]
        v_p = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2) # [bs, h, len_v, d_k]

        # 2. Content-Content Attention Score (A) = Q * K^T
        # [bs, h, len_q, len_k]
        content_score = torch.matmul(q_p, k_p.transpose(-2, -1)) 

        # 3. Relative Positional Attention Score (B)
        # Tạo ma trận khoảng cách tương đối
        # range_vec_q = torch.arange(len_q)
        # range_vec_k = torch.arange(len_k)
        # distance_mat = range_vec_k[None, :] - range_vec_q[:, None] # [len_q, len_k]
        # distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        # final_mat = distance_mat_clipped + self.max_relative_position # [len_q, len_k] (indices)
        
        # Để tối ưu tốc độ, dùng hàm helper hoặc tính trực tiếp
        device = q.device
        range_vec_q = torch.arange(len_q, device=device)
        range_vec_k = torch.arange(len_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Lấy embedding RPE
        # [len_q, len_k, d_k]
        rpe_k = self.relative_pe_k(final_mat) 

        # Tính Q * RPE_K^T
        # q_p: [bs, h, len_q, d_k] -> [len_q, bs*h, d_k]
        q_p_r = q_p.permute(2, 0, 1, 3).reshape(len_q, bs*self.h, self.d_k)
        # rpe_k: [len_q, len_k, d_k] -> [len_q, d_k, len_k]
        # matmul: [len_q, bs*h, d_k] x [len_q, d_k, len_k] -> [len_q, bs*h, len_k]
        relative_score = torch.matmul(q_p_r, rpe_k.transpose(1, 2))
        # Reshape lại: [len_q, bs, h, len_k] -> [bs, h, len_q, len_k]
        relative_score = relative_score.view(len_q, bs, self.h, len_k).permute(1, 2, 0, 3)

        # 4. Tổng hợp Score
        scores = (content_score + relative_score) / math.sqrt(self.d_k)

        # 5. Masking & Softmax
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        
        attn = functional.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # 6. Output = Attn * V
        output = torch.matmul(attn, v_p) # [bs, h, len_q, d_k]
        
        # Concat heads
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        final_output = self.out(concat)
        
        return final_output, attn

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, internal_activation=functional.relu, dropout=0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.internal_activation = internal_activation
    
    def forward(self, x):
        x = self.dropout(self.internal_activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x