---
title: Agentic RL
category: concepts
tags: [ml, reinforcement-learning, agent, llm]
aliases: [Agentic Reinforcement Learning, RL for LLM Agents]
sources: [conversation-2026-04-19]
summary: 通过强化学习训练AI Agent自主规划、推理和执行复杂任务，核心包括长周期决策、过程奖励模型和分层动作。
provenance:
  extracted: 0.75
  inferred: 0.25
  ambiguous: 0.00
created: 2026-04-19T10:40:00Z
updated: 2026-04-19T10:40:00Z
---

# Agentic RL

Agentic RL（智能体强化学习）是指通过强化学习赋予 AI Agent（智能体）自主规划、推理和执行复杂任务的能力的研究方向。当前重点在于训练大语言模型（LLM）作为 Agent 完成复杂推理和工具调用任务。

## 核心研究问题

| 问题 | 描述 |
|------|------|
| **Long-Horizon RL** | 解决长周期任务（数百到数千步决策链） |
| **Process Reward Model** | 对每个中间步骤打分，而非仅对最终结果打分 |
| **Hierarchical RL** | 高层规划 + 低层执行的分层动作空间 |
| **RL for LLM Agents** | 用 RL 训练 LLM 自主工具调用、计划执行、自我纠错 |
| **Self-Improvement** | 智能体通过与环境交互持续自我改进 |

## 短周期 vs 长周期决策

| 类型 | 特点 | 奖励来源 |
|------|------|----------|
| **短周期** | 每步立即获得奖励反馈 | Outcome Reward（最终结果） |
| **长周期** | 仅最终有奖励，中间每步需引导 | Process Reward（过程奖励） |

传统 RL（如 Atari 游戏）每步都有即时奖励，但复杂任务（如软件工程、长期推理）只在任务结束时才有奖励，导致学习信号稀疏。 ^[inferred]

## Process Reward Model (PRM)

传统 RL 只奖励最终结果（Outcome Reward），但长任务中每个中间步骤都做对才能成功。PRM 对每步打分：

```python
# PRM 评估中间步骤质量
step_score = PRM(state, action, reasoning_trace)

# 而非仅在任务结束时
final_reward = env.step(final_action)
```

- 例如：代码生成任务中，PRM 评估"这个函数是否正确"
- 避免 reward hacking：模型找到取巧方式获得高分但任务实际失败

## 核心训练范式

```
LLM Agent + RL → 学会自主工具调用、计划执行、自我纠错
```

代表性工作：
- **SWE-TRACE**（2026）：用 RL 优化软件工程智能体的推理过程，通过过程奖励模型评估每个代码编辑步骤
- **UniDoc-RL**（2026）：粗到细的视觉 RAG + 分层动作 + 密集奖励

## 关键概念

### PASS@k 分析

衡量 RL 扩展能力的方式：
- **PASS@k**：k 次尝试中至少成功 1 次的概率
- 用于评估 RL 是否扩展了 LLM Agent 的能力边界

### Rubric Process Reward

基于评分准则的过程奖励，根据详细标准对每步推理打分。 ^[inferred]

## 为什么重要

| 传统 RL | Agentic RL |
|---------|-----------|
| 固定环境 | 开放世界、与工具/API 交互 |
| 短决策链 | 长时序、多阶段任务 |
| 人工设计奖励 | PRM 自动过程评估 |
| 单步技能 | 自主规划 + 工具使用 |

## 应用场景

- **自动化代码开发**：AI 自主完成复杂编程任务
- **多步骤推理**：数学证明、复杂查询
- **机器人控制**：长期任务规划
- **多智能体协作**：多个 Agent 协同工作

## 关联页面

- [[concepts/transformer-architecture]] — 底层模型架构
- [[concepts/multi-head-attention]] — 注意力机制
- [[references/attention-is-all-you-need]] — Transformer 原始论文
