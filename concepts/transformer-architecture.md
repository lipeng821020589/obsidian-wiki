---
title: Transformer Architecture
category: concepts
tags: [ml, nlp, architecture, deep-learning]
aliases: [Transformer, transformer, seq2seq]
sources: [conversation-2026-04-19]
summary: 完全基于Attention机制的序列建模架构，抛弃RNN/CNN并行训练长序列的核心突破。
provenance:
  extracted: 0.85
  inferred: 0.15
  ambiguous: 0.00
created: 2026-04-19T10:30:00Z
updated: 2026-04-19T10:30:00Z
---

# Transformer Architecture

Transformer 是 2017 年《Attention Is All You Need》提出的完全基于 Attention 机制的序列建模架构，摒弃了 RNN 和 CNN，实现了并行训练和长距离依赖的直接建模。

## 核心架构

### Encoder-Decoder 结构

```
Input → Embedding + Positional Encoding
        ↓
    [Encoder Layer] × N
        ├── Multi-Head Self-Attention
        ├── Add & Norm
        ├── Feed Forward
        └── Add & Norm
        ↓
    [Decoder Layer] × N
        ├── Masked Multi-Head Self-Attention
        ├── Cross-Attention (Encoder-Decoder)
        ├── Feed Forward
        └── Add & Norm
        ↓
    Linear + Softmax → Output
```

### 关键组件

#### 1. Scaled Dot-Product Attention

```python
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

- **√d_k 缩放因子**：防止点积值过大导致 softmax 梯度消失
- 每个位置可以"看到"所有其他位置的信息

#### 2. Multi-Head Attention

```python
head_i = Attention(QW_i^q, KW_i^k, VW_i^v)
MultiHead = Concat(head_1, ..., head_h) · W^O
```

- 将 Q/K/V 投影到 h 个子空间（h=8, d_k=64）
- 每个头学习不同类型的依赖关系
- 总维度保持 d_model = 512 不变，参数量与单头 Attention 等价

#### 3. Positional Encoding

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Transformer 本身不感知位置，通过 PE 注入顺序信息
- sin/cos 形式支持相对位置学习

#### 4. Feed Forward Layer

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

- 两层线性变换，中间 ReLU
- 参数量约占整个模型的 2/3

#### 5. Layer Normalization + Residual Connection

每个子层：`LayerNorm(x + Sublayer(x))`

- 残差连接确保梯度流动
- LayerNorm 稳定训练

### Decoder 的 Masked Attention

训练时使用 Masked Self-Attention，防止看到未来 token：

- 第 1 个输出只能看第 1 个输入
- 第 2 个输出只能看第 1、2 个输入
- 以此类推

## 关键设计决策

| 设计 | 选择 | 原因 |
|------|------|------|
| Scaled Dot-Product | 除以 √d_k | 防止点积过大，softmax 梯度消失 |
| Multi-Head | h=8, d_k=64 | 多子空间并行学习不同依赖 |
| 残差连接 | 每子层都有 | 深层网络训练的保障 |
| Post-LN（原版） | LayerNorm 在残差之后 | 后续研究证明 Pre-LN 更稳定 |

## 优缺点

**优点：**
- 完全并行化（O(1) 链式依赖 vs RNN 的 O(n)）
- 长距离依赖直接建模
- 可解释性强（可视化 Attention 权重）

**缺点：**
- 复杂度 O(n²)，长序列计算量大
- 位置编码是人工设计的，非学习得到
- 对短序列训练效率不如 CNN

## 演进脉络

- [[references/attention-is-all-you-need]] — 原始论文
- [[concepts/multi-head-attention]] — 核心组件详解
- [[concepts/flashattention]] — IO 优化版本
- BERT / GPT 系列 — 基于 Transformer 的预训练模型

## Sources

- [[references/attention-is-all-you-need]]
