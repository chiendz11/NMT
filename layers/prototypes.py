import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import math
import logging

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=200, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self._max_seq_length = max_seq_length
        
        # Tạo ma trận vị trí (Positional Encoding Matrix)
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                div_term = math.exp(i * -(math.log(10000.0) / d_model))
                pe[pos, i] = math.sin(pos * div_term)
                pe[pos, i+1] = math.cos(pos * div_term)
                
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [FIX] Kiểm tra độ dài và cắt nếu input dài hơn max_seq_length
        if(x.size(1) > self._max_seq_length):
            # logging.warning(f"Input length {x.size(1)} > max {self._max_seq_length}. Trimming.")
            x = x[:, :self._max_seq_length]
        
        x = x * math.sqrt(self.d_model)
        
        # Cộng vị trí vào embedding
        # Slice trực tiếp pe theo độ dài hiện tại của x
        x = x + self.pe[:, :x.size(1)]
        
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
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
        
        # Sửa lỗi contiguous để tránh warning hoặc lỗi view
        concat = value.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        return output, attn

    def attention(self, q, k, v, mask=None, dropout=None):
        """Calculate the attention and output the attention & value"""
    
        # scores: [Batch, Head, Q_len, K_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # --- [SYNC FIX] TỰ ĐỘNG CẮT MASK NẾU DỮ LIỆU BỊ CẮT ---
            # Nếu mask dài hơn scores (do input bị cắt ở PositionalEncoder), ta cắt mask theo.
            
            # Cắt chiều Key (Cột cuối cùng)
            k_len = scores.size(-1) 
            if mask.size(-1) > k_len:
                mask = mask[..., :k_len]
            
            # Cắt chiều Query (nếu mask là dạng ma trận vuông [Batch, Len, Len])
            q_len = scores.size(-2)
            if mask.dim() > 2 and mask.size(-2) > q_len:
                mask = mask[..., :q_len, :]
            # ------------------------------------------------------

            mask = mask.unsqueeze(1) # add a dimension to account for head
            
            # [CRITICAL FIX FOR FP16]
            # Thay đổi từ -1e9 thành -1e4 (-10000).
            # Giá trị này đủ nhỏ để Softmax ra 0, nhưng nằm trong vùng an toàn của FP16 (-65504).
            scores = scores.masked_fill(mask==0, -1e4)
            
        scores = functional.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores, v)
        return output, scores

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        # Tạo parameters
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