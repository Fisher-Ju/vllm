import torch
import torch.nn as nn
import math


## 手撕transformer代码
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        ## vocab_size:token的词表长度
        ## d_model:映射成几维的向量
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):  ## 乘以d_model
        return self.embedding(x) * math.sqrt(self.d_model)

    def to(self, device):
        self.embedding = self.embedding.to(device)


class PositionalEncoding(nn.Module):  ##位置编码
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.max_len = max_len
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, x.size(1)]


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    return torch.matmul(attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        query = transform(query, self.q)
        key = transform(key, self.k)
        value = transform(value, self.v)

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.linear_out(x)

    def transform(self, query, key, value, mask=None):
        return self.dropout(attention(query, key, value, mask))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        return self.dropout(sub_layer(self.norm(x))) + x



class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = nn.LayerNorm(d_model)
        self.subLayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, out1, mask):
        out1 = self.subLayers[0](out1, lambda x: self.self_attn(x, x, x, mask))
        out2 = self.subLayers[1](out1, self.feed_forward(out1))
        return out2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.subLayers = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, x, memory, src_mask, tgt_mask):
        out1 = self.subLayers[0](x, lambda y1: self.self_attn(x, x, x, tgt_mask))
        out2 = self.subLayers[1](out1, lambda y2: self.cross_attn(out1, memory, memory, src_mask))
        out3 = self.subLayers[2](out2, self.feed_forward(out2))

        return out3


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dmodel=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, dmodel),
            PositionalEncoding(dmodel)
        )

        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab, dmodel),
            PositionalEncoding(dmodel)
        )

        attn = lambda: MultiHeadAttention(d_model=dmodel, num_heads=h, dropout=dropout)
        ff = lambda: FeedForward(d_model=dmodel, d_ff=d_ff, dropout=dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(dmodel, attn, ff, dropout) for _ in range(N)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(dmodel, attn, ff, dropout) for _ in range(N)
        ])
        self.out = nn.Linear(dmodel, tgt_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask=None):
        src = self.src_embed(src)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.out(out)
