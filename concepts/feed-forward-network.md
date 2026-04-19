---
title: Feed-Forward Network (FFN)
category: concepts
tags: [ml, nlp, transformer, architecture]
aliases: [FFN, Position-wise FFN, Feed-Forward]
sources: [conversation-2026-04-19]
summary: FFN是Transformer中占比约2/3的关键组件，对每个token独立做两层线性变换+非线性激活，增强单个token表征。
provenance:
  extracted: 0.82
  inferred: 0.18
  ambiguous: 0.00
created: 2026-04-19T11:10:00Z
updated: 2026-04-19T11:10:00Z
---

# Feed-Forward Network (FFN)

FFN（Feed-Forward Network）是 Transformer Encoder 和 Decoder 中每个 Layer 的第二个子层，独立作用于每个 position，与 Self-Attention 形成互补。

## 数学形式

```python
FFN(x) = max(0, x · W₁ + b₁) · W₂ + b₂
```

展开：
```
x ∈ ℝ^(d_model)       # 输入向量
h = x · W₁ + b₁        # 第一层线性变换  (d_model → d_ff)
h = ReLU(h)            # 非线性激活
o = h · W₂ + b₂        # 第二层线性变换  (d_ff → d_model)
```

## 维度与参数量

原论文设置：d_model=512, d_ff=2048（扩展比 4x）

| 参数矩阵 | 形状 | 参数量 |
|----------|------|--------|
| W₁ | (512, 2048) | 1,048,576 |
| b₁ | (2048,) | 2,048 |
| W₂ | (2048, 512) | 1,048,576 |
| b₂ | (512,) | 512 |
| **总计** | | **2,099,200 ≈ 2.1M** |

> FFN 占整个 Transformer 约 **2/3 的参数量**，是模型参数的主体。

## FFN 与 Self-Attention 的互补

| 组件 | 作用 | 范围 |
|------|------|------|
| **Self-Attention** | 建模 token 间关系 | 跨 position 交互 |
| **FFN** | 转换单个 token 表征 | Position-wise 独立 |

FFN 可理解为 **key-value memory**：第一层是"寻址"，第二层是"读取值"。 ^[inferred]

## 激活函数演进

| 激活函数 | 公式 | 使用模型 |
|----------|------|----------|
| **ReLU** | max(0, x) | 原版 Transformer |
| **GELU** | x·Φ(x) | GPT, BERT, T5 |
| **SwiGLU** | Swish(x)·σ(x) | LLaMA, PaLM |

## FFN 变体

### MoE (Mixture of Experts)

用稀疏 MoE 层替换 FFN：
- 总参数量 = n 个 FFN（如 8 个）
- 每次推理只激活 top-k（如 top-2）
- 代表：Mixtral, LLaMA-MoE, DeepSeek-MoE

### LoRA (Low-Rank Adaptation)

通过低秩分解减少 FFN 可训练参数：
```
W₁ ≈ A · B   (r << d_ff)
```

## FFN 参数量占比（GPT-3 175B）

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Self-Attention (Q/K/V/O) | ~630M | ~3.6% |
| **FFN (W₁/W₂)** | **~1.4B** | **~80%** |
| Embedding | ~49M | ~0.3% |

## 关联页面

- [[concepts/transformer-architecture]] — Transformer 整体架构
- [[concepts/multi-head-attention]] — Self-Attention 机制
