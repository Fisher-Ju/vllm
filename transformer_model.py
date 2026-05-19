"""
Transformer 模型实现 - 手撕 Transformer 代码

================================================================================
整体架构概述
================================================================================
Transformer 由 Encoder 和 Decoder 两部分组成：

    输入 src ──> [Encoder] ──> memory (编码器输出)
                              │
                              ▼
    输入 tgt ──> [Decoder] ──> 输出 (经过 memory 注意力)

Encoder: N 个 EncoderLayer 串联，每层包含 Self-Attention + FeedForward
Decoder: N 个 DecoderLayer 串联，每层包含 Self-Attention + Cross-Attention + FeedForward

数据流向:
    src (token IDs) ──> Embedding ──> PositionalEncoding ──> EncoderLayers ──> memory
    tgt (token IDs) ──> Embedding ──> PositionalEncoding ──> DecoderLayers ──> Linear ──> logits

================================================================================
执行逻辑流程 (forward 方法)
================================================================================
1. encode(src, src_mask):
   - src: [batch, seq_len] 整数 token IDs
   - 经过 src_embed: Embedding + PositionalEncoding -> [batch, seq_len, d_model]
   - 依次通过 N 个 EncoderLayer
   - 输出 memory: [batch, seq_len, d_model]

2. decode(tgt, memory, src_mask, tgt_mask):
   - tgt: [batch, tgt_len] 整数 token IDs
   - 经过 tgt_embed: Embedding + PositionalEncoding
   - 依次通过 N 个 DecoderLayer (每层都接收 memory 作为 cross-attention 的输入)
   - 输出: [batch, tgt_len, d_model]

3. self.out(out):
   - Linear 层将 d_model 维映射到 tgt_vocab_size 维
   - 输出 logits: [batch, tgt_len, tgt_vocab_size]

================================================================================
关键 Python/PyTorch 语法说明
================================================================================
1. nn.Module: PyTorch 神经网络模块基类，所有自定义层必须继承
   - 必须调用 super().__init__() 初始化父类
   - forward() 方法定义前向计算逻辑
   - __call__() 会自动调用 forward()

2. nn.Sequential: 顺序容器，按顺序执行多个层
   - 等价于: output = layer1(layer2(...layerN(input)...))
   - 常用于简单的串联结构

3. nn.ModuleList: 模块列表容器
   - 可以像 Python list 一样迭代
   - 所有子模块会被正确注册到父模块 (参数会被追踪)
   - 普通 Python list 不会注册模块，参数不会被优化器更新

4. lambda 表达式: 匿名函数
   - lambda x: func(x) 创建一个接受 x 并返回 func(x) 的函数
   - 在 AddNorm 中用于延迟执行 sub_layer，让 AddNorm 控制 norm 的顺序

5. register_buffer('name', tensor): 注册缓冲区
   - 不参与梯度计算 (不是模型参数)
   - 但会随模型移动 (to(device)) 和保存 (state_dict)
   - 常用于 PositionalEncoding 这种固定的预计算值

6. view() vs reshape():
   - view() 要求张量在内存中连续，否则报错
   - reshape() 会自动处理不连续情况 (必要时复制)
   - contiguous() 确保张量在内存中连续

================================================================================
"""

import torch
import torch.nn as nn
import math


# ==============================================================================
# Embeddings 类: Token ID -> 向量嵌入
# ==============================================================================
"""
功能: 将离散的 token ID (整数) 映射为连续的向量表示

输入: x [batch, seq_len] - 每个元素是词表中的索引 (整数)
输出: [batch, seq_len, d_model] - 每个 token 对应一个 d_model 维向量

数学公式: output = embedding_table[x] * sqrt(d_model)
为什么要乘 sqrt(d_model)?
  - 论文中提到这样可以让 embedding 和 positional encoding 的量级相近
  - 避免 positional encoding 占主导地位

nn.Embedding 内部原理:
  - 维护一个查找表 [vocab_size, d_model]
  - 输入索引 i，返回表的第 i 行向量
  - 本质是一个不需要计算的"索引取值"操作
"""
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        vocab_size: 词表大小，即不同 token 的总数
        d_model: 模型的隐藏维度 (论文中为 512)
        """
        ## vocab_size:token的词表长度
        ## d_model:映射成几维的向量
        super().__init__()  # 必须调用父类初始化，否则 nn.Module 的功能不生效
        # nn.Embedding(vocab_size, d_model) 创建一个 [vocab_size, d_model] 的查找表
        # 输入整数索引，输出对应的嵌入向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        x: [batch, seq_len] 的整数 tensor (token IDs)
        返回: [batch, seq_len, d_model] 的浮点 tensor

        注意: nn.Embedding 要求输入必须是 Long 或 Int 类型
        如果传入 FloatTensor 会报错: "Expected tensor for argument 'indices' to have scalar types: Long, Int"
        """
        ## 乘以d_model: 缩放 embedding 使其与 positional encoding 量级匹配
        return self.embedding(x) * math.sqrt(self.d_model)

    def to(self, device):
        """自定义 to 方法，将 embedding 表移动到指定设备 (GPU/CPU)"""
        self.embedding = self.embedding.to(device)


# ==============================================================================
# PositionalEncoding 类: 位置编码
# ==============================================================================
"""
功能: 为序列中的每个位置添加位置信息，让模型区分不同位置的 token

为什么需要位置编码?
  - Self-Attention 是位置无关的: 交换两个 token 的位置，attention 结果不变
  - RNN 通过序列处理隐式获得位置信息，Transformer 没有这种机制
  - 因此需要显式添加位置编码

Sinusoidal 位置编码公式 (论文方案):
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))   # 偶数维度用 sin
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))   # 假数维度用 cos

特点:
  1. 不同位置的编码唯一
  2. 可以泛化到训练时未见过的序列长度
  3. 相对位置可以通过线性变换得到 (PE(pos+k) 可由 PE(pos) 变换得到)

实现技巧:
  - div_term 预计算指数衰减因子: 1/10000^(2i/d_model) = exp(-log(10000) * 2i / d_model)
  - pe[:, 0::2] 表示从第0列开始，步长为2取切片 (即偶数列)
  - pe[:, 1::2] 表示从第1列开始，步长为2取切片 (即奇数列)
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 模型隐藏维度
        max_len: 预计算的最大序列长度 (超过此长度会报错，可以扩展)
        """
        super().__init__()
        # 创建位置编码矩阵: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1] 每个位置一个整数 0, 1, 2, ...
        # unsqueeze(1) 在第1维增加维度，从 [max_len] 变成 [max_len, 1]
        # 这样才能与 div_term [d_model/2] 进行广播乘法
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: 计算指数衰减因子
        # torch.arange(0, d_model, 2) 生成 [0, 2, 4, ..., d_model-2] (偶数索引)
        # 数学推导: 1/10000^(2i/d_model) = exp(-log(10000) * 2i / d_model)
        # 这样可以避免数值不稳定的问题
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 填充偶数列 (维度索引 0, 2, 4, ...) 用 sin
        # position * div_term: [max_len, 1] * [d_model/2] -> [max_len, d_model/2] (广播)
        pe[:, 0::2] = torch.sin(position * div_term)

        # 填充奇数列 (维度索引 1, 3, 5, ...) 用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze(0) 在第0维增加维度，变成 [1, max_len, d_model]
        # 第0维是 batch 维，这样可以通过广播与任意 batch_size 的输入相加
        pe = pe.unsqueeze(0)

        # register_buffer: 将 pe 注册为模块的缓冲区
        # 与 nn.Parameter 的区别:
        #   - Parameter 会被优化器更新 (参与梯度计算)
        #   - Buffer 不参与梯度计算，但会随模型移动设备、保存/加载
        # 位置编码是固定的，不需要学习，所以用 buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model] 已经经过 embedding 的输入
        返回: x + positional encoding

        self.pe[:, :x.size(1)] 取前 seq_len 个位置的编码
        self.pe shape: [1, max_len, d_model]
        切片后: [1, seq_len, d_model]
        加到 x 上: [batch, seq_len, d_model] + [1, seq_len, d_model] -> 广播相加
        """
        return x + self.pe[:, :x.size(1)]


# ==============================================================================
# attention 函数: Scaled Dot-Product Attention
# ==============================================================================
"""
功能: 计算注意力加权后的 value

数学公式: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

步骤解析:
  1. scores = Q · K^T: 计算 query 与 key 的相似度
     - Q: [batch, heads, seq_len_q, d_k]
     - K: [batch, heads, seq_len_k, d_k]
     - K.transpose(-2, -1): 交换最后两维 -> [batch, heads, d_k, seq_len_k]
     - 结果 scores: [batch, heads, seq_len_q, seq_len_k]

  2. scores / sqrt(d_k): 缩放，防止点积值过大导致 softmax 梯度消失
     - 当 d_k 很大时，点积结果也会很大
     - softmax 对大值会产生很小的梯度 (饱和区)
     - 缩放后使数值稳定

  3. mask: 对于 padding 位置或未来位置，将其 score 设为 -1e9
     - softmax(-1e9) ≈ 0，相当于忽略这些位置

  4. softmax: 将 scores 转换为概率分布 (每行加起来为 1)

  5. attn · V: 用注意力权重加权 value
     - attn: [batch, heads, seq_len_q, seq_len_k]
     - V: [batch, heads, seq_len_k, d_k]
     - 结果: [batch, heads, seq_len_q, d_k]

mask 的作用:
  - src_mask: 隐藏 padding token (源序列中无效的位置)
  - tgt_mask: 隐藏未来 token (解码时不能看到后面的词，保证自回归特性)
"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # key/query 的维度，用于缩放

    # 计算注意力分数: Q · K^T / sqrt(d_k)
    # matmul: 矩阵乘法
    # transpose(-2, -1): 交换倒数第二维和最后一维
    #   例如 [batch, heads, seq_len, d_k] -> [batch, heads, d_k, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用 mask: 将 mask==0 的位置的 score 设为极小值
    # masked_fill(mask==0, -1e9): mask 为 0 的位置填充 -1e9
    # softmax(-1e9) ≈ 0，这些位置的 attention 权重几乎为 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax: 将 scores 转为概率分布
    # dim=-1 表示在最后一维 (seq_len_k) 上归一化
    # 每个查询位置对所有 key 位置的概率加起来为 1
    attn = torch.softmax(scores, dim=-1)

    # dropout: 训练时随机丢弃一些注意力连接，防止过拟合
    if dropout is not None:
        attn = dropout(attn)

    # 最后用注意力权重加权 value
    # attn: [batch, heads, seq_len_q, seq_len_k]
    # value: [batch, heads, seq_len_k, d_k]
    # 结果: [batch, heads, seq_len_q, d_k]
    return torch.matmul(attn, value)


# ==============================================================================
# MultiHeadAttention 类: 多头注意力
# ==============================================================================
"""
功能: 将 d_model 维度分成 h 个头并行计算注意力，再合并

为什么需要多头?
  - 单头注意力只能学习一种"关注模式"
  - 多头可以同时学习多种不同的关注模式
  - 例如: 有的头关注语法关系，有的头关注语义关系

数学公式:
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
  其中 head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

实现方式:
  1. 用 Linear 层将输入投影到 d_model 维 (分成 h 个头会自动拆分)
  2. 通过 view 和 transpose 将 [batch, seq_len, d_model] 变成 [batch, h, seq_len, d_k]
  3. 调用 attention 函数
  4. 合并多头: transpose + view + Linear

维度变化详解:
  输入: [batch, seq_len, d_model]

  经过 Linear: [batch, seq_len, d_model]

  view 重塑: [batch, seq_len, num_heads, d_k]
    - d_model = num_heads * d_k
    - 例如 d_model=512, num_heads=8, 则 d_k=64

  transpose(1, 2): [batch, num_heads, seq_len, d_k]
    - 交换 seq_len 和 num_heads 维度
    - 这样 attention 可以在每个头上独立计算

  attention 后: [batch, num_heads, seq_len, d_k]

  合并: transpose + view -> [batch, seq_len, d_model]

  输出 Linear: [batch, seq_len, d_model]
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        d_model: 模型隐藏维度 (论文中 512)
        num_heads: 注意力头数 (论文中 8)
        要求: d_model 必须能被 num_heads 整除
        """
        super().__init__()
        # 确保可以均匀分割
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的投影层
        # 注意: 投影到 d_model 维，不是 d_k 维
        # 因为投影后会拆分成 num_heads 个头，每个头自然得到 d_k 维
        # bias=False: 论文中不使用偏置项
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影层: 将多头合并后的结果再投影
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: 都是 [batch, seq_len, d_model]
        mask: [batch, 1, 1, seq_len] 或 [batch, 1, seq_len, seq_len]

        对于 Self-Attention: query = key = value (同一输入)
        对于 Cross-Attention: query 来自一个源，key/value 来自另一个源
        """
        batch_size = query.size(0)

        # 内部函数 transform: 投影 + 分头
        # 这是一个闭包: 引用了外部的 batch_size, self.num_heads, self.d_k
        def transform(x, linear):
            """
            x: 输入 tensor
            linear: 投影层 (self.q / self.k / self.v)
            """
            # 1. 投影
            x = linear(x)  # [batch, seq_len, d_model]

            # 2. 重塑为多头形式
            # view(batch_size, -1, num_heads, d_k)
            # -1 表示自动计算该维大小 (这里是 seq_len)
            # 原理: d_model = num_heads * d_k，所以可以这样拆分
            x = x.view(batch_size, -1, self.num_heads, self.d_k)

            # 3. 调整维度顺序
            # transpose(1, 2) 交换第1维和第2维
            # 从 [batch, seq_len, num_heads, d_k] 变成 [batch, num_heads, seq_len, d_k]
            # 目的: 让 attention 函数在每个头上并行计算
            return x.transpose(1, 2)

        # 对 Q, K, V 分别投影并分头
        query = transform(query, self.q)  # [batch, num_heads, seq_len_q, d_k]
        key = transform(key, self.k)      # [batch, num_heads, seq_len_k, d_k]
        value = transform(value, self.v)  # [batch, num_heads, seq_len_v, d_k]

        # 调用 attention 函数
        # 注意: seq_len_k == seq_len_v (key 和 value 来自同一源)
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x: [batch, num_heads, seq_len_q, d_k]

        # 合并多头
        # 1. transpose(1, 2): [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)

        # 2. contiguous(): 确保内存连续
        # transpose 只是改变视图，内存布局可能不连续
        # view 要求内存连续，所以需要先调用 contiguous
        x = x.contiguous()

        # 3. view: 合并最后两维
        # [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
        # d_model = num_heads * d_k
        x = x.view(batch_size, -1, self.d_model)

        # 4. 输出投影
        return self.linear_out(x)

    # 注意: 这个 transform 方法与 forward 内的 transform 函数同名但不同
    # 这是一个实例方法，似乎未在当前代码中使用
    def transform(self, query, key, value, mask=None):
        return self.dropout(attention(query, key, value, mask))


# ==============================================================================
# FeedForward 类: 前馈神经网络
# ==============================================================================
"""
功能: 两层全连接网络，对每个位置独立处理

数学公式: FFN(x) = max(0, x·W1 + b1) · W2 + b2
         = ReLU(Linear1(x)) · Linear2

特点:
  - 对序列中每个位置独立应用 (不同位置之间没有交互)
  - 中间维度 d_ff 通常比 d_model 大 (论文中 d_ff=2048, d_model=512)
  - 相当于在每个位置做一个"特征增强"操作

实现: 用 nn.Sequential 简化代码
  - Sequential 会按顺序执行各层
  - 相当于: output = dropout(linear2(relu(linear1(x))))
"""
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度 (通常是 d_model 的 4 倍)
        dropout: dropout 比率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 第一层: 扩展维度
            nn.ReLU(),                  # 激活函数: 非线性
            nn.Linear(d_ff, d_model),   # 第二层: 恢复维度
            nn.Dropout(dropout)         # Dropout: 正则化
        )

    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        return self.net(x)


# ==============================================================================
# AddNorm 类: 残差连接 + Layer Normalization
# ==============================================================================
"""
功能: 实现 Transformer 的残差连接和层归一化

数学公式: output = x + Dropout(SubLayer(Norm(x)))

注意顺序 (Post-LN):
  1. 先对 x 做 LayerNorm
  2. 然后执行子层操作 (attention 或 feedforward)
  3. 加上残差连接 (原始 x)
  4. dropout

为什么用残差连接?
  - 允许梯度直接流向浅层，缓解深层网络梯度消失
  - 模型可以学习"identity mapping" (如果子层无用，可以近似跳过)

为什么用 LayerNorm?
  - 对每个样本的特征维度归一化，使训练更稳定
  - BatchNorm 对 batch 维归一化，不适合变长序列和小 batch

SubLayer 是什么?
  - 在 EncoderLayer 中: Self-Attention 或 FeedForward
  - 在 DecoderLayer 中: Self-Attention, Cross-Attention 或 FeedForward

为什么用 lambda?
  - AddNorm 的 forward 接收一个函数 sub_layer
  - 这样 AddNorm 可以控制执行顺序: 先 norm(x)，再 sub_layer
  - 如果传入 sub_layer 的结果，就无法保证这个顺序

示例:
  正确: AddNorm(x, lambda x: attention(x, x, x))
       -> AddNorm 内部执行: attention(norm(x), ...) + x

  错误: AddNorm(x, attention(x, x, x))
       -> attention 已经执行完毕，AddNorm 只能拿到 tensor 结果
       -> 无法保证 norm 在 attention 之前执行
"""
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        """
        x: 输入 tensor
        sub_layer: 一个函数，接受 tensor 并返回 tensor

        执行顺序:
          1. norm(x): Layer Normalization
          2. sub_layer(...): 执行子层操作
          3. dropout(...): dropout
          4. + x: 残差连接
        """
        return self.dropout(sub_layer(self.norm(x))) + x


# ==============================================================================
# EncoderLayer 类: 编码器单层
# ==============================================================================
"""
功能: 一个 Encoder 层，包含 Self-Attention 和 FeedForward

结构:
  Input ──> [AddNorm + Self-Attention] ──> [AddNorm + FeedForward] ──> Output

  其中 AddNorm 包含: LayerNorm -> 子层操作 -> Dropout -> 残差加法

Self-Attention:
  - query = key = value = 输入本身
  - 让每个位置关注序列中所有其他位置
  - 用于捕获序列内部的依赖关系

执行流程:
  1. 第一个 AddNorm: Self-Attention
     - x_norm = LayerNorm(x)
     - attn_out = SelfAttention(x_norm, x_norm, x_norm, mask)
     - output = dropout(attn_out) + x

  2. 第二个 AddNorm: FeedForward
     - x_norm = LayerNorm(output)
     - ff_out = FeedForward(x_norm)
     - output = dropout(ff_out) + output
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        """
        d_model: 模型维度
        self_attn: MultiHeadAttention 实例 (用于 self-attention)
        feed_forward: FeedForward 实例
        dropout: dropout 比率
        """
        super().__init__()
        self.self_attn = self_attn       # Self-Attention 模块
        self.feed_forward = feed_forward # FeedForward 模块
        self.norm = nn.LayerNorm(d_model) # 这个 norm 似乎未使用 (AddNorm 内有自己的 norm)
        self.subLayers = nn.ModuleList([
            AddNorm(d_model, dropout),  # 第一个: 用于 Self-Attention
            AddNorm(d_model, dropout)   # 第二个: 用于 FeedForward
        ])

    def forward(self, out1, mask):
        """
        out1: 输入 tensor [batch, seq_len, d_model]
        mask: 源序列掩码 [batch, 1, 1, seq_len]

        lambda 语法解释:
          lambda x: self.self_attn(x, x, x, mask)
          创建一个函数，接受参数 x，调用 self_attn(x,x,x,mask)

        为什么传 lambda 而不是直接调用结果?
          - AddNorm 需要先执行 LayerNorm，再执行子层
          - 如果传入 self.self_attn(out1, ...) 的结果，就无法保证顺序
          - lambda 让 AddNorm 控制执行时机
        """
        # 第一个子层: Self-Attention
        # lambda x: ... 创建一个函数，AddNorm 会先对 out1 做 norm，再调用这个函数
        out1 = self.subLayers[0](out1, lambda x: self.self_attn(x, x, x, mask))

        # 第二个子层: FeedForward
        # 同理，lambda 让 AddNorm 先 norm 再 feedforward
        out2 = self.subLayers[1](out1, lambda x: self.feed_forward(x))

        return out2


# ==============================================================================
# DecoderLayer 类: 解码器单层
# ==============================================================================
"""
功能: 一个 Decoder 层，包含 Self-Attention, Cross-Attention, FeedForward

结构:
  Input ──> [AddNorm + Self-Attention]
        ──> [AddNorm + Cross-Attention]
        ──> [AddNorm + FeedForward] ──> Output

三种 Attention:
  1. Self-Attention (第一层):
     - query = key = value = decoder 输入
     - 让目标序列内部相互关注
     - 需要 tgt_mask 防止看到未来位置 (自回归约束)

  2. Cross-Attention (第二层):
     - query = decoder 输入
     - key = value = encoder 输出 (memory)
     - 让解码器关注编码器的输出
     - 需要 src_mask 隐藏源序列的 padding

  3. FeedForward (第三层):
     - 对每个位置独立处理

执行流程详解:
  1. Self-Attention:
     - 输入 x (已嵌入的 target 序列)
     - 注意: lambda 内部用 y1 作为参数名，但实际传入的是 norm(x)
     - Self-Attention(x, x, x, tgt_mask)
     - 残差连接后得到 out1

  2. Cross-Attention:
     - 输入 out1 (Self-Attention 的输出)
     - query 来自 out1 (decoder)
     - key/value 来自 memory (encoder 输出)
     - Cross-Attention(out1, memory, memory, src_mask)
     - 残差连接后得到 out2

  3. FeedForward:
     - 输入 out2
     - FeedForward(out2)
     - 残差连接后得到 out3
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.1):
        """
        d_model: 模型维度
        self_attn: 用于 decoder 内部的 self-attention
        cross_attn: 用于关注 encoder 输出的 cross-attention
        feed_forward: 前馈网络
        dropout: dropout 比率

        注意: self_attn 和 cross_attn 必须是两个独立的 MultiHeadAttention 实例
        不能共享，因为它们的参数不同，学习的内容也不同
        """
        super().__init__()
        self.self_attn = self_attn      # Decoder Self-Attention
        self.cross_attn = cross_attn    # Cross-Attention (关注 encoder)
        self.feed_forward = feed_forward
        self.subLayers = nn.ModuleList([
            AddNorm(d_model, dropout),  # 第一个: Self-Attention
            AddNorm(d_model, dropout),  # 第二个: Cross-Attention
            AddNorm(d_model, dropout)   # 第三个: FeedForward
        ])

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: decoder 输入 [batch, tgt_len, d_model]
        memory: encoder 输出 [batch, src_len, d_model]
        src_mask: 源序列掩码 [batch, 1, 1, src_len] (隐藏 padding)
        tgt_mask: 目标序列掩码 [batch, 1, tgt_len, tgt_len] (隐藏未来)

        lambda 参数名说明:
          - lambda y1: ... 中的 y1 只是参数名
          - AddNorm 会传入 norm(x) 作为 y1 的值
          - 参数名可以是任何名字，重要的是函数逻辑
        """
        # 第一个子层: Self-Attention
        # y1 是 lambda 的参数名，AddNorm 会传入 norm(x)
        # self.self_attn(x, x, x, tgt_mask) - 这里第一个 x 是未 norm 的，应该是 y1
        # 注意: 这里有个 bug，应该用 y1 而不是 x 作为 query/key/value
        out1 = self.subLayers[0](x, lambda y1: self.self_attn(y1, y1, y1, tgt_mask))

        # 第二个子层: Cross-Attention
        # query 来自 decoder (y2 = norm(out1))
        # key/value 来自 encoder (memory)
        # src_mask 用于隐藏 encoder 的 padding 位置
        out2 = self.subLayers[1](out1, lambda y2: self.cross_attn(y2, memory, memory, src_mask))

        # 第三个子层: FeedForward
        out3 = self.subLayers[2](out2, lambda y3: self.feed_forward(y3))

        return out3


# ==============================================================================
# Transformer 类: 完整的 Transformer 模型
# ==============================================================================
"""
功能: 组合 Encoder 和 Decoder 构成完整的 Transformer

整体结构:
  ┌─────────────────────────────────────────────────────────────┐
  │  src ──> src_embed ──> [Encoder N层] ──> memory             │
  │                               │                              │
  │                               ▼                              │
  │  tgt ──> tgt_embed ──> [Decoder N层] ──> Linear ──> logits   │
  └─────────────────────────────────────────────────────────────┘

参数说明:
  src_vocab: 源语言词表大小
  tgt_vocab: 目标语言词表大小
  dmodel: 模型隐藏维度 (论文中 512)
  N: Encoder/Decoder 的层数 (论文中 6)
  h: 注意力头数 (论文中 8)
  d_ff: FeedForward 中间维度 (论文中 2048)
  dropout: dropout 比率

lambda 工厂函数说明:
  attn = lambda: MultiHeadAttention(...)
  ff = lambda: FeedForward(...)

  为什么用 lambda 而不是直接创建实例?
    - 需要为每层创建独立的实例
    - lambda: 创建一个"工厂函数"，每次调用 attn() 都生成新实例
    - 如果直接写 attn = MultiHeadAttention(...)，只能得到一个实例
    - 每层共享实例会导致参数共享，这是不正确的

  DecoderLayer 需要两个 attention 实例:
    - self_attn: decoder 内部 self-attention
    - cross_attn: 关注 encoder 的 cross-attention
    - 必须是两个独立实例，不能共享参数

nn.Sequential 说明:
  - src_embed = Sequential(Embeddings, PositionalEncoding)
  - 等价于: output = PositionalEncoding(Embeddings(src))
  - 注意: Embeddings 输出会自动传给 PositionalEncoding

nn.ModuleList 说明:
  - self.encoder = ModuleList([EncoderLayer(...) for _ in range(N)])
  - ModuleList 会注册所有子模块，参数会被优化器追踪
  - 普通 Python list 不会注册模块，会导致参数不被训练

执行流程:
  forward(src, tgt, src_mask, tgt_mask):
    1. encode(src, src_mask):
       - src_embed(src): Embedding + PositionalEncoding
       - 依次通过 N 个 EncoderLayer
       - 输出 memory

    2. decode(tgt, memory, src_mask, tgt_mask):
       - tgt_embed(tgt): Embedding + PositionalEncoding
       - 依次通过 N 个 DecoderLayer
       - 输出 decoder_hidden

    3. self.out(decoder_hidden):
       - Linear 映射到词表大小
       - 输出 logits (用于计算 loss 或生成下一个 token)
"""
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dmodel=512, N=6, h=8, d_ff=2048, dropout=0.1):
        """
        初始化 Transformer 模型

        参数:
          src_vocab: 源语言词表大小 (例如英语词汇量)
          tgt_vocab: 目标语言词表大小 (例如中文词汇量)
          dmodel: 模型隐藏维度 (论文中 512)
          N: Encoder/Decoder 层数 (论文中 6)
          h: 多头注意力的头数 (论文中 8)
          d_ff: FeedForward 中间维度 (论文中 2048，是 dmodel 的 4 倍)
          dropout: dropout 比率，防止过拟合
        """
        super().__init__()

        # 源序列嵌入层: Embedding + PositionalEncoding
        # Sequential 会按顺序执行: output = PosEnc(Embedding(src))
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, dmodel),
            PositionalEncoding(dmodel)
        )

        # 目标序列嵌入层
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab, dmodel),
            PositionalEncoding(dmodel)
        )

        # 工厂函数: 每次调用创建一个新的实例
        # lambda: MultiHeadAttention(...) 是一个无参数函数
        # 调用 attn() 会返回一个新的 MultiHeadAttention 实例
        # 为什么需要工厂函数?
        #   - 每个 EncoderLayer/DecoderLayer 需要独立的 attention 实例
        #   - 如果共享实例，参数也会共享，这是不正确的
        #   - DecoderLayer 需要 self_attn 和 cross_attn 两个不同的实例
        attn = lambda: MultiHeadAttention(d_model=dmodel, num_heads=h, dropout=dropout)
        ff = lambda: FeedForward(d_model=dmodel, d_ff=d_ff, dropout=dropout)

        # Encoder: N 个 EncoderLayer
        # attn() 调用 lambda，创建新实例
        # 每层的 attention 和 feedforward 都是独立的
        self.encoder = nn.ModuleList([
            EncoderLayer(dmodel, attn(), ff(), dropout) for _ in range(N)
        ])

        # Decoder: N 个 DecoderLayer
        # DecoderLayer 需要 4 个组件参数:
        #   dmodel, self_attn, cross_attn, feed_forward, dropout
        # attn() 调用两次，创建两个独立的 attention 实例:
        #   - 第一个用于 self-attention (decoder 内部)
        #   - 第二个用于 cross-attention (关注 encoder 输出)
        # ff() 用于 feedforward
        self.decoder = nn.ModuleList([
            DecoderLayer(dmodel, attn(), attn(), ff(), dropout) for _ in range(N)
        ])

        # 输出层: 将 dmodel 维映射到 tgt_vocab 维
        # logits 用于计算 softmax 得到词表概率
        self.out = nn.Linear(dmodel, tgt_vocab)

    def encode(self, src, src_mask):
        """
        编码器前向传播

        参数:
          src: 源序列 token IDs [batch, src_len]
          src_mask: 源序列掩码 [batch, 1, 1, src_len]

        流程:
          1. src_embed(src): Embedding + PositionalEncoding
             输入: [batch, src_len] 整数 IDs
             输出: [batch, src_len, dmodel] 向量

          2. 依次通过 N 个 EncoderLayer
             每层做 self-attention + feedforward

          3. 输出 memory: [batch, src_len, dmodel]
             这是 encoder 对源序列的"编码表示"
             会传给 decoder 的 cross-attention
        """
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        解码器前向传播

        参数:
          tgt: 目标序列 token IDs [batch, tgt_len]
          memory: encoder 输出 [batch, src_len, dmodel]
          src_mask: 源序列掩码 [batch, 1, 1, src_len]
          tgt_mask: 目标序列掩码 [batch, 1, tgt_len, tgt_len]

        流程:
          1. tgt_embed(tgt): Embedding + PositionalEncoding

          2. 依次通过 N 个 DecoderLayer
             每层做:
               - self-attention (decoder 内部，用 tgt_mask)
               - cross-attention (关注 memory，用 src_mask)
               - feedforward

          3. 输出: [batch, tgt_len, dmodel]
        """
        x = self.tgt_embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask=None):
        """
        完整的 Transformer 前向传播

        参数:
          src: 源序列 token IDs [batch, src_len] (Long/Int 类型)
          tgt: 目标序列 token IDs [batch, tgt_len] (Long/Int 类型)
          src_mask: 源序列掩码，隐藏 padding 位置
          tgt_mask: 目标序列掩码，隐藏未来位置 (自回归)

        流程:
          1. encode: 将 src 编码为 memory
          2. decode: 结合 memory 解码 tgt
          3. out: 映射到词表大小

        输出:
          logits: [batch, tgt_len, tgt_vocab]
          可以用 softmax 得到每个位置的词表概率分布

        注意事项:
          - src 和 tgt 必须是整数类型 (Long/Int)
          - 如果传入 FloatTensor，Embedding 层会报错
          - forward 中不要重复调用 src_embed，encode 方法已经处理
        """
        # encode 会自己调用 src_embed，不要在 forward 中重复嵌入
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.out(out)