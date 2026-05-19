# run_demo.py
import torch
# 从同目录下的模型文件导入Transformer类
from transformer_model import Transformer

if __name__ == '__main__':
    # 超参数
    src_vocab_size = 4
    tgt_vocab_size = 4
    batch_size = 2
    src_seq_len = 1
    tgt_seq_len = 1

    # 初始化模型
    model = Transformer(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        dmodel=2,
        N=1,
        h=1,
        d_ff=2
    )

    # 构造假输入
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 全 1 padding 掩码：须与 scores (batch, num_heads, Lq, Lk) 广播对齐，故为 (B, 1, 1, L)
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    tgt_mask = torch.ones(batch_size, 1, 1, tgt_seq_len)

    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)

    print(f"输入src shape: {src.shape}, src={src}")
    print(f"模型输出shape: {output.shape}, output={output}")