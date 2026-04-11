# 大模型量化开源项目笔记

> 生成日期: 2026-04-11
> 主题: LLM Quantization Open Source Projects

## 目录

- [[#推理框架]]
- [[#量化库]]
- [[#工具集]]

---

## 推理框架

### 1. llama.cpp

| 属性 | 内容 |
|------|------|
| **组织** | ggml-org |
| **GitHub** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **语言** | C/C++ |
| **License** | MIT |

#### 核心特性

- **纯CPU推理**: 专为CPU优化，无GPU也可运行
- **GGUF格式**: 自研量化模型存储格式
- **Apple Silicon**: 优先支持Metal
- **混合精度**: K-Quants支持不同层不同精度

#### 量化方法

| 类别 | 格式 | 说明 |
|------|------|------|
| **Legacy** | Q2_K - Q5_K | 经典量化 |
| **K-Quants** | IQ1_S - IQ5_K | 混合精度，更高质量 |
| **重要矩阵** | imatrix | 校准数据驱动优化 |

#### 性能表现

```
Model: LLaMA 3.1 8B
Hardware: M2 Pro (Apple Silicon)

Format   | Size   | Prompt/s | Gen/s
---------|--------|----------|-------
FP16     | 16.0 GiB | 772    | 86
Q4_K_M   | 4.9 GiB  | 772    | 77
Q5_K_M   | 5.9 GiB  | 768    | 80
Q8_0     | 9.0 GiB  | 741    | 74
```

#### 适用场景

- 本地/边缘设备部署
- 无GPU环境
- 快速原型验证
- macOS设备

#### 生态

- [llama.cpp](https://github.com/ggerganov/llama.cpp) 主仓库
- [Ollama](https://github.com/ollama/ollama) - 上层包装
- [llamafile](https://github.com/Mozilla-Ocho/llamafile) - 单文件分发
- [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) - 在线量化

---

### 2. vLLM

| 属性 | 内容 |
|------|------|
| **组织** | vllm-project |
| **GitHub** | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **语言** | Python |
| **License** | Apache 2.0 |

#### 核心特性

- **PagedAttention**: 业界领先的高吞吐注意力机制
- **Continuous Batching**: 动态批处理
- **FlashAttention**: 高效注意力kernel
- **KV Cache**: 高效管理

#### 量化支持

| 量化类型 | 说明 | 状态 |
|----------|------|------|
| **FP8** | MXFP8/MXFP4, NVFP4 | ✅ |
| **INT8/INT4** | 动态量化 | ✅ |
| **GPTQ/AWQ** | 权重量化 | ✅ |
| **GGUF** | llama.cpp格式 | ✅ |
| **compressed-tensors** | 压缩张量 | ✅ |

#### 性能表现

> 官方基准 (A100-80GB)

```
Model: Llama-2-70B
Concurrency: 32

Method    | Throughput (req/s) | TPS
----------|---------------------|-------
HuggingFace| 45                | 2,850
vLLM       | 142               | 8,950
Speedup    | ~3.2x             | ~3.1x
```

#### 适用场景

- 高吞吐服务部署
- 生产环境
- 多模型服务
- 分布式推理

#### 生态

- 200+ 模型架构支持
- 支持NVIDIA/AMD/Intel GPU
- OpenAI兼容API
- Tensor/Pipeline并行

#### 特色功能

- Speculative decoding
- Prefill/decode分离
- Multi-LoRA
- 200+开源模型原生支持

---

### 3. Text Generation Inference (TGI)

| 属性 | 内容 |
|------|------|
| **组织** | Hugging Face |
| **GitHub** | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| **语言** | Rust + Python |
| **License** | Apache 2.0 |

#### 核心特性

- **Continuous Batching**: 高效请求批处理
- **Quantization**: 支持bitsandbytes
- **Prefix Caching**: 前缀缓存
- **Speculative Decoding**: 预测解码

#### 适用场景

- Hugging Face生态集成
- 生产级部署
- 微服务架构

---

## 量化库

### 1. bitsandbytes

| 属性 | 内容 |
|------|------|
| **组织** | bitsandbytes-foundation |
| **GitHub** | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **语言** | Python/CUDA |
| **License** | MIT |

#### 核心功能

| 功能 | 描述 | 量化精度 |
|------|------|----------|
| **LLM.int8()** | 推理量化，无性能损失 | 8-bit |
| **QLoRA** | 4-bit量化微调 | 4-bit |
| **8-bit Optimizers** | 高效训练 | 8-bit |

#### 技术细节

- **LLM.int8()**: 向量化量化 + 离群值处理
- **NF4**: 4-bit NormalFloat (信息论最优)
- **双重量化**: 量化常数再量化

#### 平台支持

| 平台 | LLM.int8() | QLoRA | 8-bit Optimizer |
|------|------------|-------|-----------------|
| Linux x86 CPU | ✅ | ✅ | ✅ |
| NVIDIA GPU | ✅ | ✅ | ✅ |
| AMD GPU | ✅ | ✅ | ✅ |
| Intel GPU | ✅ | ✅ | ✅ |
| Apple Silicon | ✅ | ✅ | ❌ |
| Windows | ✅ | ✅ | ✅ |

#### 性能

```
Model: Llama-2-70B

Method    | GPU Memory | Relative
----------|-----------|-----------
FP16      | 140 GB    | 1.0x
8-bit     | 70 GB     | 0.5x
4-bit     | 35 GB     | 0.25x
```

- 无性能损失 (vs FP16)
- 支持70B+模型

#### 使用示例

```python
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb

# 8-bit推理
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_8bit=True
)

# QLoRA微调 (4-bit)
# 需要使用bitsandbytes的LoRA模块
```

#### 生态集成

- 🤗 Transformers原生支持
- 🤗 Diffusers支持
- 🤗 PEFT支持

---

### 2. AutoGPTQ

| 属性 | 内容 |
|------|------|
| **组织** | AutoGPTQ |
| **GitHub** | [AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) |
| **语言** | Python |
| **License** | Apache 2.0 |

#### 核心特性

- **GPTQ算法实现**: 基于ICLR 2023论文
- **CUDA优化**: 高效GPU推理
- **ORT集成**: 支持ONNX Runtime
- **TRT集成**: 支持TensorRT

#### 性能

```
Model: Llama-2-70B
Hardware: A100-80GB

Bits | ppl (wikitext) | Memory
-----|---------------|--------
16   | 2.42         | 140 GB
8    | 2.58         | 70 GB
4    | 3.12         | 35 GB
```

#### 与bitsandbytes对比

| 特性 | AutoGPTQ | bitsandbytes |
|------|----------|--------------|
| 主算法 | GPTQ | 多种 |
| 训练支持 | ❌ (仅推理) | ✅ |
| 4-bit | ✅ | ✅ |
| 8-bit | ✅ (推理) | ✅ |
| ONNX | ✅ | ❌ |
| TensorRT | ✅ | ❌ |

#### 适用场景

- 生产环境部署
- 需要TensorRT/ONNX优化
- 纯推理场景

---

### 3. AWQ (llm-awq)

| 属性 | 内容 |
|------|------|
| **组织** | mit-han-lab |
| **GitHub** | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) |
| **License** | Apache 2.0 |

#### 核心特性

- **原生AWQ实现**: MLSys 2024最佳论文
- **M3 EP**: 混合专家MoE支持
- **TinyChat**: 高效推理框架
- **4-bit权重量化**: 仅权重

#### 性能

```
Model: Llama-2-70B
Benchmark: MMLU

Method    | MMLU acc | Memory
----------|----------|-------
FP16      | 68.2%    | 140 GB
AWQ 4-bit | 67.1%    | 35 GB
Loss      | -1.1%    | -75%
```

#### 适用场景

- 边缘设备部署
- 需要保护显著权重
- 移动端推理

---

## 工具集

### 1. lm-evaluation-harness

| 属性 | 内容 |
|------|------|
| **GitHub** | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |

#### 用途

- 标准化LLM评估
- 支持200+基准
- 量化模型评估

#### 量化评估

```bash
lm_eval --model gptq \
    --model_args pretrained=<path>,bit_quality=4 \
    --tasks mmlu
```

---

### 2. lmdeploy

| 属性 | 内容 |
|------|------|
| **组织** | OpenMMLab |
| **GitHub** | [OpenMMLab/lmdeploy](https://github.com/OpenMMLab/lmdeploy) |

#### 核心特性

- **TurboMind**: 自研推理引擎
- **PagedAttention**: 高效显存管理
- **多模态**: 支��VLM

---

## 对比总结

### 推理框架对比

| 框架 | 优化重点 | CPU支持 | GPU吞吐 | 量化格式 |
|------|----------|---------|---------|----------|
| **llama.cpp** | 本地/边缘 | ✅ 优先 | 一般 | GGUF原生 |
| **vLLM** | 生产服务 | ❌ | 最高 | 多种 |
| **TGI** | HuggingFace | ❌ | 高 | 多种 |

### 量化库对比

| 库 | 量化类型 | 训练支持 | 推理 | 生态 |
|------|----------|---------|------|------|
| **bitsandbytes** | 8-bit, 4-bit | ✅ QLoRA | ✅ | HuggingFace |
| **AutoGPTQ** | 4-bit, 3-bit | ❌ | ✅ | ONNX/TensorRT |
| **llm-awq** | 4-bit | ❌ | ✅ | TinyChat |

### 性能推荐

| 场景 | 推荐方案 |
|------|----------|
| **本地CPU运行** | llama.cpp + GGUF Q4_K_M |
| **高吞吐服务** | vLLM + FP8 |
| **4-bit微调** | bitsandbytes QLoRA |
| **生产推理(ONNX)** | AutoGPTQ + ONNX Runtime |
| **边缘设备** | AWQ + TinyChat |

---

## 快速开始

### llama.cpp (本地运行)

```bash
# 1. 量化模型
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# 2. 运行
./llama-cli -m model-q4_k_m.gguf -n 256
```

### vLLM (服务)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", quantization="AWQ")
# 自动使用量化权重
```

### bitsandbytes (微调)

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb

# 4-bit加载
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

# 添加LoRA
model = get_peft_model(model, config)
```

---

## 引用

```bibtex
@software{llamacpp2024,
  title = {llama.cpp},
  author = {Georgi Gerganov},
  url = {https://github.com/ggerganov/llama.cpp},
  year = {2024}
}

@software{vllm2024,
  title = {vLLM},
  author = {Woosuk Kwon and Zhuohan Li and others},
  url = {https://github.com/vllm-project/vllm},
  year = {2024}
}

@software{bitsandbytes2024,
  title = {bitsandbytes},
  author = {Tim Dettmers and others},
  url = {https://github.com/bitsandbytes-foundation/bitsandbytes},
  year = {2024}
}
```

---

## Tags

#llm-quantization #inference #llamacpp #vllm #bitsandbytes #awq #gptq