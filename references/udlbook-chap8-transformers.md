# Chap 8: Transformers

> UDLbook Chapter 8 精读笔记
>
> **官方资源**: [GitHub Notebooks/Chap08](https://github.com/udlbook/udlbook/tree/main/Notebooks/Chap08)
>
> **关联阅读**: [[transformer-paper-deep-read]] — Attention Is All You Need 论文精读

---

## 1. Transformer 架构总览

### 1.1 核心思想

Transformer 完全基于注意力机制，摒弃了 RNN 的循环结构和 CNN 的卷积结构：

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer                              │
├─────────────────────────────────────────────────────────────┤
│  Input → [Embedding + Positional Encoding]                   │
│         ↓                                                    │
│  Encoder: [Multi-Head Self-Attention → FFN] × N              │
│         ↓                                                    │
│  Decoder: [Masked Self-Attention → Cross-Attention → FFN] × N │
│         ↓                                                    │
│  Output → [Linear + Softmax]                                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 与 RNN/CNN 的对比

| 特性 | RNN | CNN | **Transformer** |
|------|-----|-----|-----------------|
| 路径长度 | O(n) | O(log n) | **O(1)** |
| 可并行化 | 低 | 中 | **高** |
| 长距离依赖 | 难 | 难 | **易** |
| 位置感知 | 隐式 | 需编码 | **需编码** |

---

## 2. Scaled Dot-Product Attention

### 2.1 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2.2 为什么除以 $\sqrt{d_k}$？

**数学推导**：

假设 $q$ 和 $k$ 的分量独立、均值0、方差1：
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

则 $\text{Var}(q \cdot k) = d_k$

当 $d_k$ 大时，点积的方差变大，softmax 进入饱和区（梯度接近0）。

**解决**：缩放因子 $\frac{1}{\sqrt{d_k}}$，使方差恢复到1。

### 2.3 PyTorch 实现

```python
# ▶ Scaled Dot-Product Attention
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)
    K: (batch, heads, seq_len, d_k)
    V: (batch, heads, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算点积
    scores = Q @ K.transpose(-2, -1)  # (batch, heads, n, m)
    
    # 2. 缩放
    scores = scores / math.sqrt(d_k)
    
    # 3. 应用掩码（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和
    output = attn_weights @ V
    
    return output, attn_weights

# 测试
batch, heads, seq_len, d_k, d_v = 2, 8, 10, 64, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_v)

out, weights = scaled_dot_product_attention(Q, K, V)
print(f"输出 shape: {out.shape}")  # (2, 8, 10, 64)
print(f"Attention weights shape: {weights.shape}")  # (2, 8, 10, 10)
```

---

## 3. Multi-Head Attention

### 3.1 核心思想

在不同的表示子空间中并行学习注意力：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3.2 PyTorch 实现

```python
# ▶ Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # 线性投影
        self.W_Q = nn.Linear(d_model, num_heads * self.d_k)
        self.W_K = nn.Linear(d_model, num_heads * self.d_k)
        self.W_V = nn.Linear(d_model, num_heads * self.d_v)
        self.W_O = nn.Linear(num_heads * self.d_v, d_model)
    
    def split_heads(self, x):
        # (batch, seq, h*d) -> (batch, heads, seq, d)
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.h, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch = Q.shape[0]
        
        # 线性投影 + 分头
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))
        
        # Scaled Dot-Product Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, -1, self.h * self.d_v)
        
        # 最终线性投影
        output = self.W_O(attn_output)
        
        return output

# 测试
d_model, num_heads = 512, 8
mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(2, 10, d_model)
out = mha(x, x, x)
print(f"输出 shape: {out.shape}")  # (2, 10, 512)
```

---

## 4. 三种 Attention 用途

### 4.1 表格总结

| 位置 | Attention 类型 | Query 来源 | Key/Value 来源 |
|------|---------------|-----------|----------------|
| Encoder | Self-Attention | 同一层 Encoder | 同一层 Encoder |
| Decoder | Masked Self-Attention | 同一层 Decoder | 同一层 Decoder |
| Decoder | Cross-Attention | Decoder 上一层 | Encoder 最终输出 |

### 4.2 Masked Self-Attention（因果掩码）

```python
# ▶ 因果掩码实现
def create_causal_mask(size):
    """创建下三角掩码"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # True 表示可attention

# 应用
mask = create_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
```

---

## 5. Position-wise Feed-Forward Network

### 5.1 公式

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

### 5.2 代码

```python
# ▶ FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# d_ff 通常是 d_model 的 4 倍
ffn = FeedForward(d_model=512, d_ff=2048)
```

---

## 6. Positional Encoding

### 6.1 正弦/余弦编码

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 6.2 PyTorch 实现

```python
# ▶ Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 6.3 可视化

```python
# ▶ 可视化位置编码
import matplotlib.pyplot as plt

d_model, max_len = 64, 200
pe = PositionalEncoding(d_model, max_len).pe[0].numpy()

plt.figure(figsize=(12, 6))
plt.imshow(pe.T, aspect='auto', cmap='RdBu_r')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding (d_model=64, max_len=200)')
plt.colorbar()
plt.show()
```

---

## 7. 完整 Encoder Layer

```python
# ▶ Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention + Residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN + Residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

# 测试
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)
out = encoder_layer(x)
print(f"Encoder Layer 输出: {out.shape}")  # (2, 10, 512)
```

---

## 8. 完整 Decoder Layer

```python
# ▶ Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-Attention (Query from Decoder, Key/Value from Encoder)
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
```

---

## 9. 完整 Transformer

```python
# ▶ 完整 Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Encoder 和 Decoder 堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 源序列编码
        src = self.dropout(self.pos_encoding(self.src_embedding(src)))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # 目标序列解码
        tgt = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        # 输出
        return self.fc_out(tgt)
```

---

## 10. Wiki 关联

| 主题 | 链接 |
|------|------|
| Transformer 原始论文 | [[transformer-paper-deep-read]] |
| Attention 数学基础 | [[7_应用_Attention机制]] |
| 优化器 | [[udlbook-chap3-optimization]] |
| 神经网络基础 | [[udlbook-chap4-neural-networks]] |

---

## Tags

#transformer #attention #self-attention #multi-head #positional-encoding #nlp #deep-learning
