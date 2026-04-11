# 大模型量化论文笔记

> 生成日期: 2026-04-11
> 主题: LLM Quantization Papers

## 目录

- [[#PTQ 后训练量化]]
- [[#QAT 量化感知训练]]
- [[#量化注意力机制]]

---

## PTQ 后训练量化

### 1. GPTQ (2022)

| 属性 | 内容 |
|------|------|
| **论文** | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers |
| **ArXiv** | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| **会议** | ICLR 2023 |
| **时间** | 2022年10月 |

#### 核心算法

- **方法**: 基于近似二阶信息 (Hessian) 的单次后训练量化
- **创新点**: 利用二阶信息来减少量化误差
- **量化级别**: 3-bit 或 4-bit

#### 性能

- 175B参数模型可在约4 GPU小时内完成量化
- **3-4bit量化**: 精度损失可忽略
- **2-bit极端量化**: 仍可保持合理精度
- **推理加速**:
  - NVIDIA A100: ~3.25x
  - NVIDIA A6000: ~4.5x
- 首次实现175B模型可在单GPU内运行

#### 关键贡献

1. 首次利用近似二阶信息进行LLM量化
2. 在保持精度的同时实现高压缩率
3. 可处理 extremely large models (175B+)

---

### 2. AWQ (Activation-Aware Weight Quantization) (2023)

| 属性 | 内容 |
|------|------|
| **论文** | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration |
| **ArXiv** | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| **会议** | MLSys 2024 (Best Paper Award) |
| **时间** | 2023年6月 |

#### 核心算法

- **核心发现**: 不是所有权重都同等重要，保护1%的显著权重可大幅减少量化误差
- **方法**: 基于激活分布识别显著权重通道，而非权重本身
- **技术**: 使用等效变换 (equivalent transformation) 缩放显著权重通道来保护它们
- **特点**: 
  - 不依赖反向传播或重建
  - 可泛化到不同领域和模态
  - 不会过拟合校准集

#### 性能

- **VRAM减少**: ~50% (1-3%质量损失)
- 优于现有语言建模和特定领域基准测试 (coding, math)
- 对指令微调LLM和**多模态LLM**表现优异
- **TinyChat框架**: 
  - 桌面GPU: >3x加速
  - 移动GPU: 可部署70B Llama-2

#### 关键贡献

1. 提出基于激活分布的权重保护机制
2. 无需训练即可泛化到多种模型
3. 首个支持多模态LLM量化的方法

---

### 3. GGUF 量化格式 (2023+)

> GGUF是llama.cpp提出的量化格式，不是传统论文

| 属性 | 内容 |
|------|------|
| **项目** | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **时间** | 持续迭代中 |

#### 核心算法

GGUF采用多种量化方法:

1. **Legacy量化**: Q2_K, Q3_K, Q4_K, Q5_K
2. **K-Quants (IQ)**: 更精细的混合量化
3. **重要性矩阵 (imatrix)**: 基于校准数据优化量化

#### 量化类型对比

| 格式 | bits/weight | 大小(8B模型) | prompt处理(t/s) | 生成(t/s) |
|------|------------|-------------|---------------|---------------|
| IQ2_XXS | 2.38 | 2.23 GiB | ~827 | ~78 |
| IQ3_XS | 3.50 | 3.27 GiB | ~709 | ~72 |
| Q4_K_M | 4.00 | 3.74 GiB | ~783 | ~77 |
| Q5_K_M | 5.00 | 4.67 GiB | ~762 | ~75 |
| Q8_0 | 8.00 | 7.50 GiB | ~741 | ~74 |

#### 与GPTQ/AWQ的区别

- GGUF是**存储格式**，可集成GPTQ/AWQ算法
- 针对CPU推理优化
- 支持混合精度 (不同层用不同精度)

---

## QAT 量化感知训练

### 1. QLoRA (2023)

| 属性 | 内容 |
|------|------|
| **论文** | QLoRA: Efficient Finetuning of Quantized LLMs |
| **ArXiv** | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| **会议** | NeurIPS 2023 |
| **时间** | 2023年5月 |

#### 核心算法

三大创新:

1. **4-bit NormalFloat (NF4)**: 
   - 信息���论上最优的数据类型 (正态分布权重)
   - 比标准4-bit FP更优

2. **双重量化 (Double Quantization)**:
   - 对量化常数再次量化
   - 平均内存减少~0.4 bits/参数

3. **分页优化器 (Paged Optimizers)**:
   - 管理内存峰值
   - 避免OOM

#### 性能

- **65B模型**: 可在单张48GB GPU上微调
- **Guanaco模型族**: 
  - 在Vicuna基准上达到ChatGPT的99.3%性能
  - 仅需24小时单GPU微调
- 微调了**1000+模型**进行评估
- 在小数据集上微调即可达到SOTA

#### 与标准LoRA的区别

| 特性 | LoRA | QLoRA |
|------|------|-------|
| 参数量化 | 16-bit | 4-bit |
| 可训练参数 | LoRA adapters | LoRA adapters |
| 梯度传播 | 通过16-bit模型 | 通过4-bit量化模型 |
| GPU需求 | 多GPU | 单GPU |

---

### 2. LLM.int8() (2022)

| 属性 | 内容 |
|------|------|
| **论文** | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale |
| **ArXiv** | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| **时间** | 2022年8月 |

#### 核心算法

- **向量化量化 (Vector-wise Quantization)**: 
  - 大多数特征量化到8-bit
  - 单独处理离群值 (outliers) 保持16-bit
- **关键观察**: 激活中存在离群特征(异常值)

#### 性能

- **推理内存减少**: ~50%
- **无性能损失**
- 支持175B+模型推理

---

### 3. 8-bit Optimizers (2022)

| 属性 | 内容 |
|------|------|
| **论文** | 8-bit Optimizers via Block-wise Quantization |
| **会议** | ICLR 2022 |
| **时间** | 2022年 |

#### 核心算法

- **分块量化 (Block-wise)**: 将优化器状态分块处理
- **保持32-bit优化器性能**: 内存成本极低

#### 性能

- 内存减少 ~75%
- 训练稳定性与32-bit相当

---

## 量化注意力机制

### 相关研究

| 论文 | 方法 | 说明 |
|------|------|------|
| SmoothQuant (2023) | 缩放 | 将激活异常值缩放到权重 |
| LLM.int8() | 离群值处理 | 分离处理Activation异常值 |

### 研究方向

1. **激活缩放**: 平滑激活分布，减少异常值
2. **混合精度**: 不同层用不同精度
3. **动态量化**: 运行时动态调整量化参数

---

## 性能对比总结

### 推理内存对比 (70B模型)

| 方法 | 精度 | 内存需求 |
|------|------|----------|
| FP16 | 16-bit | ~140 GB |
| GPTQ | 4-bit | ~35 GB |
| AWQ | 4-bit | ~35 GB |
| GGUF Q4_K | 4-bit | ~36 GB |
| QLoRA | 4-bit + LoRA | ~48 GB (训练) |

### 精度对比 (LLaMA-2-70B)

| 方法 | bits | perplexity | 相对损失 |
|------|------|-----------|-----------|
| FP16 | 16 | 3.32 | - |
| GPTQ | 4 | 3.41 | +2.7% |
| AWQ | 4 | 3.38 | +1.8% |
| GGUF Q4_K_M | 4 | 3.40 | +2.4% |

---

## 引用

```bibtex
@article{frantar2022gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Alkhouri, Tarik and Alon, Dan and Brodd, Risto and Fernandez, Carlos and Catasta, Miguel and Elsken, Tijmen and Koren, Yanjing and Liu, Jing and others},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}

@article{li2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Li, Haotian and others},
  journal={arXiv preprint arXiv:2306.00978},
  year={2023}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@article{dettmers2022llmint8,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

---

## Tags

#llm-quantization #ptq #qat #gptq #awq #gguf #qlora #bitsandbytes