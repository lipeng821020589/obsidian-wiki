---
title: Multi-Head Attention
category: concepts
tags: [ml, nlp, attention, transformer]
aliases: [MHA, MultiHead]
sources: [conversation-2026-04-19]
summary: Transformer的核心组件，8个并行Attention头各学不同依赖模式（语法/语义/位置），参数量与单头等价。
provenance:
  extracted: 0.80
  inferred: 0.20
  ambiguous: 0.00
created: 2026-04-19T10:35:00Z
updated: 2026-04-19T10:35:00Z
---

# Multi-Head Attention

Multi-Head Attention（MHA）是 Transformer 的核心创新，通过并行多个独立的 Attention 头，使模型能够在不同子空间中同时学习不同类型的依赖关系。

## 为什么需要多头？

单头 Attention 的局限：
- **表达能力有限**：单一空间无法捕获多种类型的依赖
- **信息丢失**：复杂语义关系无法被单一表示完全捕获

多头通过将 Q/K/V 投影到多个子空间，让每个头独立学习，从而克服上述问题。 ^[inferred]

## 数学推导

### Step 1：线性投影

```
Q = X · Wq      # (n, d_model) → (n, d_k)
K = X · Wk      # (n, d_model) → (n, d_k)
V = X · Wv      # (n, d_model) → (n, d_v)
```

### Step 2：分头（Reshape）

```
Q.shape = (n, d_model) → split → (h, n, d_k)  # h=8, d_k=64
```

### Step 3：各头独立计算 Attention

```
head_i = Attention(Q_i · W_i^q, K_i · W_i^k, V_i · W_i^v)
       = softmax(Q_i K_iᵀ / √d_k) · V_i
```

### Step 4：拼接 + 输出投影

```
MultiHead = Concat(head_1, ..., head_h) · W^O
          # shape: (n, h × d_v) = (n, d_model)
```

## 完整前向过程

```
输入 X (seq_len, d_model=512)
  │
  ├── Linear(Wq) → Q
  ├── Linear(Wk) → K  ← 并行投影
  └── Linear(Wv) → V
  │
  ├── Reshape: (512) → (8 × 64)
  │
  ├── H0: Q₀K₀ᵀ/√64 → softmax → head₀
  ├── H1: Q₁K₁ᵀ/√64 → softmax → head₁
  ├── H2: ...
  ├── H3: ...
  ├── H4: ...
  ├── H5: ...
  ├── H6: ...
  ├── H7: ...
  │
  ├── Concat: (8 × 64) → (512)
  └── Linear(W⁰) → Output
  │
  └── 与输入 X 残差连接 + LayerNorm
```

## 各头可能学到的模式

| Head | 模式 | 可能的语义含义 |
|------|------|--------------|
| H0 | Local-Left | 紧邻左侧 token（主谓/修饰关系） |
| H1 | Local-Right | 紧邻右侧 token |
| H2 | Diagonal | 均匀对角——位置编码等价表示 |
| H3 | CLS/Global-First | 全局信息聚合到首个 token |
| H4 | Global-Last | 聚合到序列末尾 |
| H5 | Sparse | 稀疏跳跃——指代消解 |
| H6 | Wide-Spread | 广泛分布——长距离依赖 |
| H7 | Periodic | 周期结构——句法树模式 |

## 参数量分析

| 参数矩阵 | 形状 | 参数量 |
|----------|------|--------|
| Wq | (d_model, d_k) | 512 × 64 = **32,768** |
| Wk | (d_model, d_k) | 512 × 64 = **32,768** |
| Wv | (d_model, d_v) | 512 × 64 = **32,768** |
| W⁰ | (h×d_v, d_model) | 8×64 × 512 = **262,144** |
| **总计** | | **~360K** |

> 注意：MHA 的参数量与单头 Attention **完全相同**，多头只是分计算而非增加参数。

## PyTorch 实现框架

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_k = d_model // n_heads  # 64
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.Wq(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N, D)
        return self.Wo(attn_output), attn_weights
```

## 现代变体

| 变体 | 核心改进 |
|------|---------|
| **FlashAttention** | IO-aware 优化，减少 HBM 访问 |
| **Multi-Query Attention (MQA)** | K/V 在所有 head 间共享，节省内存 |
| **Grouped-Query Attention (GQA)** | MQA 泛化，多组头共享 K/V |
| **Sparse Attention** | BigBird/Longformer，部分位置计算 |
| **Linear Attention** | Performer，用核函数近似避免 O(n²) |

## 关联页面

- [[concepts/transformer-architecture]] — Transformer 整体架构
- [[concepts/flashattention]] — FlashAttention IO 优化
