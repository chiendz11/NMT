import torch.nn as nn
from torchtext import data
import copy
import layers as layers
import torch

# =============================================================================
# CÁC LỚP GỐC (DÙNG CHO TRANSFORMER THƯỜNG)
# =============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """An layer of the encoder. Contain a self-attention accepting padding mask"""
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        self.attn = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = layers.FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm_1(x)
        x_sa, sa = self.attn(x2, x2, x2, src_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, sa

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """An layer of the decoder."""
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        self.norm_3 = layers.Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = layers.FeedForward(d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x_sa, sa = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        x_na, na = self.attn_2(x2, memory, memory, src_mask)
        x = x + self.dropout_2(x_na)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, (sa, na)

def get_clones(module, N, keep_module=True):
    if(keep_module and N >= 1):
        return nn.ModuleList([module] + [copy.deepcopy(module) for i in range(N-1)]) 
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    """Standard Encoder with Absolute Positional Encoding"""
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = layers.PositionalEncoder(d_model, dropout=dropout, max_seq_length=max_seq_length)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = layers.Norm(d_model)
        self._max_seq_length = max_seq_length

    def forward(self, src, src_mask, output_attention=False, seq_length_check=False):
        if(seq_length_check and src.shape[1] > self._max_seq_length):
            src = src[:, :self._max_seq_length]
            src_mask = src_mask[:, :, :self._max_seq_length]
        
        x = self.embed(src)
        x = self.pe(x)
        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, src_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)

class Decoder(nn.Module):
    """Standard Decoder with Absolute Positional Encoding"""
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = layers.PositionalEncoder(d_model, dropout=dropout, max_seq_length=max_seq_length)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = layers.Norm(d_model)
        self._max_seq_length = max_seq_length

    def forward(self, trg, memory, src_mask, trg_mask, output_attention=False, seq_length_check=False):
        if(seq_length_check and trg.shape[1] > self._max_seq_length):
            trg = trg[:, :self._max_seq_length]
            trg_mask = trg_mask[:, :self._max_seq_length, :self._max_seq_length]

        x = self.embed(trg)
        x = self.pe(x)

        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, memory, src_mask, trg_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)


# =============================================================================
# CÁC LỚP MỚI CHO RE-TRANSFORMER (SỬ DỤNG RELATIVE POSITIONAL ENCODING)
# =============================================================================

class ReEncoderLayer(nn.Module):
    """Encoder Layer sử dụng RelativeMultiHeadAttention"""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        
        # SỬ DỤNG RELATIVE ATTENTION (Cần đảm bảo layers.py đã có class này)
        if hasattr(layers, 'RelativeMultiHeadAttention'):
            self.attn = layers.RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        else:
            # Fallback nếu chưa update layers.py (sẽ cảnh báo hoặc gây lỗi logic)
            print("WARNING: RelativeMultiHeadAttention not found in layers.py. Using standard Attention.")
            self.attn = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
            
        self.ff = layers.FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.norm_1(x)
        x_sa, sa = self.attn(x2, x2, x2, src_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, sa

class ReDecoderLayer(nn.Module):
    """Decoder Layer sử dụng RelativeMultiHeadAttention cho Self-Attention"""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        self.norm_3 = layers.Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # Self-Attention: Dùng Relative
        if hasattr(layers, 'RelativeMultiHeadAttention'):
            self.attn_1 = layers.RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        else:
            self.attn_1 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
            
        # Cross-Attention: Vẫn dùng Standard Attention (thường không dùng RPE giữa 2 chuỗi khác nhau)
        self.attn_2 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.ff = layers.FeedForward(d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x_sa, sa = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        x_na, na = self.attn_2(x2, memory, memory, src_mask)
        x = x + self.dropout_2(x_na)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, (sa, na)

class ReEncoder(nn.Module):
    """Re-Encoder: Không dùng Absolute PE, Dùng ReEncoderLayer (có RPE)"""
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        # Không có self.pe
        
        # Sử dụng ReEncoderLayer thay vì EncoderLayer
        self.layers = get_clones(ReEncoderLayer(d_model, heads, dropout), N)
        
        self.norm = layers.Norm(d_model)
        self._max_seq_length = max_seq_length

    def forward(self, src, src_mask, output_attention=False, seq_length_check=False):
        if(seq_length_check and src.shape[1] > self._max_seq_length):
            src = src[:, :self._max_seq_length]
            src_mask = src_mask[:, :, :self._max_seq_length]

        x = self.embed(src)
        # Không cộng PE ở đây
        
        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, src_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)

class ReDecoder(nn.Module):
    """Re-Decoder: Không dùng Absolute PE, Dùng ReDecoderLayer (có RPE)"""
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        # Không có self.pe
        
        # Sử dụng ReDecoderLayer thay vì DecoderLayer
        self.layers = get_clones(ReDecoderLayer(d_model, heads, dropout), N)
        
        self.norm = layers.Norm(d_model)
        self._max_seq_length = max_seq_length

    def forward(self, trg, memory, src_mask, trg_mask, output_attention=False, seq_length_check=False):
        if(seq_length_check and trg.shape[1] > self._max_seq_length):
            trg = trg[:, :self._max_seq_length]
            trg_mask = trg_mask[:, :self._max_seq_length, :self._max_seq_length]

        x = self.embed(trg)
        # Không cộng PE ở đây

        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, memory, src_mask, trg_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)


class Config:
    """Deprecated"""
    def __init__(self):
        self.opt = {
            'train_src_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/train.en',
            'train_trg_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/train.vi',
            'valid_src_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/tst2013.en',
            'valid_trg_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/tst2013.vi',
            'src_lang':'en',
            'trg_lang':'vi',
            'max_strlen':160,
            'batchsize':1500,
            'device':'cpu',
            'd_model':512,
            'n_layers':6,
            'heads':8,
            'dropout':0.1,
            'lr':0.0001,
            'epochs':30,
            'printevery':200,
            'k':5,
            'n_warmup_steps':4000,
            'beta1':0.9,
            'beta2':0.98,
            'eps':1e-09,
            'label_smoothing':0.1,
            'save_checkpoint_epochs':5
        }
    def get(self, key, default=None):
        return self.opt.get(key, default)