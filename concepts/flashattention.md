---
title: FlashAttention
category: concepts
tags: [ml, nlp, optimization, transformer, gpu]
aliases: [FA, FlashAttention-2, IO-optimization]
sources: [conversation-2026-04-19]
summary: 通过分块计算和在线softmax，将Attention的HBM读写从O(n²)降至O(n)，让8192+长序列成为可能。
provenance:
  extracted: 0.80
  inferred: 0.20
  ambiguous: 0.00
created: 2026-04-19T10:45:00Z
updated: 2026-04-19T10:45:00Z
---

# FlashAttention

FlashAttention 是一种 IO-aware 的 Attention 加速算法，通过分块计算（Tiling）和在线归一化（Online Softmax）技术，将 Attention 的显存复杂度从 O(n²) 降至 O(n)，HBM 带宽访问量减少约 30-50 倍。

## 问题背景：标准 Attention 的瓶颈

标准 Self-Attention 的计算和显存问题：

```python
S = Q @ K^T           # O(n²) 存储 (n×n) 矩阵
P = softmax(S)         # O(n²) 存储
O = P @ V             # O(n²) 存储
```

| 问题 | 影响 |
|------|------|
| **O(n²) 显存** | 8192 context 需要 ~256MB×2 |
| **HBM 带宽瓶颈** | GPU HBM 带宽 ~1.5TB/s，但需反复读写 S/P |
| **计算 vs IO 不平衡** | 实际算力远低于峰值，IO 成为瓶颈 |

**核心矛盾：** Attention 的 IO 量（读写 VRAM）远大于计算量。

## 核心思想

> 通过分块计算（Tiling）+ 核融合（Kernel Fusion），将整个 Attention 计算融合为单个 CUDA kernel，避免反复读写 HBM。

## 在线 Softmax 技术

传统 softmax 需要全部中间结果：

```python
# 传统做法：先算完所有 exp(x_i)，再求和
total = Σ exp(x_i)
softmax_i = exp(x_i) / total
```

**在线归一化**：每个块只需存储两个标量（m=最大值，d=指数和）：

```python
# 分块增量更新
m_new = max(old_m, block_x_max)
s_block = exp(block_x - m_new)
d_new = exp(old_m - m_new) * old_d + Σ s_block
```

## 分块计算过程

```
设 Block Size = B（如 B=32）

Q = [Q₁; Q₂; ...; Q_T]    # T = n/B 块
K = [K₁; K₂; ...; K_T]
V = [V₁; V₂; ...; V_T]

对每个 Query 块 Q_i：
    1. 与所有 K 块计算 S_ij = Q_i @ K_j^T   ← 仅当前块驻留 SRAM
    2. 增量更新 softmax 统计量 (m, d)
    3. 计算对 V 的加权贡献
    4. 最终归一化得到 O_i
```

**关键：不需要存储完整的 S 和 P 矩阵！**

## IO 复杂度对比

| 对比 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| HBM 访问量 | O(n²) | O(n² / B) |
| 显存峰值 | O(n²) | O(n) |
| 速度提升 | 1x | ~3-8x |
| 8192 序列 | OOM（显存不足） | 可运行（~480ms） |

B = SRAM block size（约 32-64），HBM 访问减少约 **30-50 倍**。

## SRAM vs HBM 层次

```
GPU 架构:
┌─────────────────────────────────────┐
│            SRAM (on-chip)            │
│         ~20 MB per A100 SM           │
│      带宽: ~19 TB/s (每个 SM)        │
├─────────────────────────────────────┤
│            HBM (VRAM)                │
│         ~40-80 GB total              │
│      带宽: ~1.5 TB/s (整个 GPU)      │
└─────────────────────────────────────┘

FlashAttention: 尽量在 SRAM 里算完所有操作
                只在 block 边界读写 HBM
```

## 版本演进

| 版本 | 改进 |
|------|------|
| **FA1** | 提出分块 Attention，避免 HBM 存储 S/P |
| **FA2** | 更好并行策略：grid = (num_blocks_Q,) 而非 2D；减少冗余计算 |
| **FA3** | 支持 sequence length < block size；更好的 warp 利用 |

## 实际性能（A100 GPU）

| 序列长度 n | 标准 Attention | FlashAttention | 加速比 |
|-----------|---------------|----------------|--------|
| 512 | 14.3 ms | 3.1 ms | **4.6x** |
| 2048 | 230 ms | 35 ms | **6.6x** |
| 4096 | 920 ms | 130 ms | **7.1x** |
| 8192 | OOM | 480 ms | **∞** |

## 核心代码框架

```python
def flash_attention(Q, K, V, block_size=128):
    seq_len = Q.shape[0]
    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        Q_block = Q[i:i+block_size]         # SRAM
        m_i = torch.full((block_size,), -inf)
        d_i = torch.zeros(block_size)

        for j in range(0, seq_len, block_size):
            K_block = K[j:j+block_size]       # SRAM
            V_block = V[j:j+block_size]       # SRAM

            S_block = Q_block @ K_block.T / math.sqrt(d_k)  # SRAM 内完成

            m_block = S_block.max(dim=-1).values
            m_new = torch.maximum(m_i, m_block)
            s_block = torch.exp(S_block - m_new[:, None])
            d_new = d_i * torch.exp(m_i - m_new) + s_block.sum(dim=-1)

            P_block = s_block / d_new[:, None]
            output[i:i+block_size] += P_block @ V_block

            m_i, d_i = m_new, d_new

        output[i:i+block_size] /= d_i[:, None]

    return output
```

## 核心公式汇总

| 步骤 | 标准 | FlashAttention |
|------|------|----------------|
| **存储** | S (n²) + P (n²) | O (n)、m (n)、d (n) |
| **HBM 读写** | ~5次 n² | ~1次 n²/B |
| **显存** | O(n²) | O(n) |
| **关键技巧** | — | 在线 softmax（Scalar reduction） |

## 关联页面

- [[concepts/multi-head-attention]] — Multi-Head Attention 基础
- [[concepts/transformer-architecture]] — Transformer 整体架构
- [[concepts/agentic-rl]] — Agentic RL 相关
