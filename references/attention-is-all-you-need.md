---
title: Attention Is All You Need
category: references
tags: [paper, nlp, transformer, attention]
aliases: [Vaswani2017, Transformer Paper]
sources: [conversation-2026-04-19]
summary: 2017 NeurIPS论文，提出完全基于Attention的Transformer架构，是现代LLM的基石。
provenance:
  extracted: 0.90
  inferred: 0.10
  ambiguous: 0.00
created: 2026-04-19T10:50:00Z
updated: 2026-04-19T10:50:00Z
---

# Attention Is All You Need

- **作者**: Vaswani, Shazeer, Parmar et al. (Google Brain / Google Research)
- **发表**: NeurIPS 2017
- **引用**: ~12万次（截至2026）
- **核心思想**: 完全基于 Attention 机制，抛弃 RNN / CNN

## 核心贡献

1. 提出 Transformer 架构——完全基于 Self-Attention
2. Scaled Dot-Product Attention 机制
3. Multi-Head Attention 并行多子空间学习
4. 位置编码（Positional Encoding）注入序列顺序

## 关联页面

- [[concepts/transformer-architecture]] — 完整架构解析
- [[concepts/multi-head-attention]] — Multi-Head Attention 详解
- [[concepts/flashattention]] — 后续 IO 优化版本
