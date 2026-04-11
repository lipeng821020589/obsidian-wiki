# vLLM 深度剖析：高性能 LLM 推理引擎

> 更新时间：2026-04-11
> 来源：vLLM 官方文档、技术博客、论文

---

## 目录

1. [核心架构](#1-vllm-核心架构)
2. [推理优化技术](#2-推理优化技术)
3. [量化与部署](#3-量化与部署)
4. [API 服务](#4-api-服务)
5. [源码结构](#5-源码结构)
6. [性能对比](#6-性能对比)

---

## 1. vLLM 核心架构

### 1.1 PagedAttention 原理

**核心创新**：将操作系统虚拟内存分页机制引入注意力计算

PagedAttention 是 vLLM 的核心技术，解决了传统 LLM 推理中 KV Cache 的内存碎片化问题。传统方法预先分配连续内存块，导致：
- **内部碎片化**：为 2048 token 预留的内存可能只用到 50 token
- **外部碎片化**：即使总内存够用，单个请求仍可能因找不到足够连续内存被拒绝

```
┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention 原理                    │
├─────────────────────────────────────────────────────────────┤
│  逻辑序列 (连续)                                          │
│  [Token 1] [Token 2] [Token 3] ... [Token N]              │
│       │         │         │                │               │
│       ▼         ▼         ▼                ▼              │
│  ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐            │
│  │Block │  │Block │  │Block │  ...  │Block │            │
│  │  0   │  │  1   │  │  2   │       │  K   │            │
│  └──────┘  └──────┘  └──────┘       └──────┘            │
│       ▲         ▲         ▲                ▲               │
│       │         │         │                │               │
│  物理内存 (非连续)                                   │
│  Block 0 @ 0x1000  Block 2 @ 0x2800  Block K @ 0x5000   │
│  Block 1 @ 0x1A00  ...                                │
└─────────────────────────────────────────────────────────────┘
```

**关键优势**：
- 按需分配，碎片化仅限于最后一个块
- 无外部碎片化，所有空闲块都可利用
- 高效内存共享（beam search、parallel sampling）

**Block 大小计算**（标准 transformer）：
```
block_size = 2 (key/value) × block_size(default=16) × num_kv_heads × head_size × dtype_num_bytes
```

### 1.2 页面管理机制

```python
# KVCacheBlock 数据结构
class KVCacheBlock:
    block_id: int              # 块ID（不可变）
    block_hash: BlockHash      # 块哈希（满时分配，驱逐时重置）
    ref_cnt: int               # 引用计数
    
    # 双向链表指针（用于空闲队列）
    prev_free_block: "KVCacheBlock | None"
    next_free_block: "KVCacheBlock | None"
```

**管理组件**：
- **Block Pool**：KVCacheBlock 列表（初始化时全部分配）
- **Free Block Queue**：双向链表，管理空闲块
- **Cache Blocks**：hash → block_id 映射
- **Request Blocks**：request_id → block_ids 映射

```
┌────────────────────────────────────────┐
│         初始化后的组件                  │
├��───────────────────────────────────────┤
│  Block Pool: [B0, B1, B2, ..., B9]    │
│                                        │
│  Free Block Queue:                      │
│  HEAD ──► B0 ──> B1 ──> B2 ──> ...   │
│                                        │
│  cached_block_hash_to_block: {}        │
│  req_to_blocks: {}                     │
└────────────────────────────────────────┘
```

### 1.3 KV Cache 管理

**分配流程**：
1. 计算需要的块数：`n = ceil(new_tokens / block_size)`
2. 检查可用性
3. 从空闲队列头部取出块
4. 如果是缓存块，先驱逐

**驱逐策略 (LRU)**：
```
1. 从空闲队列头部弹出块（最久未使用）
2. 从 cache_blocks 中移除 block_id
3. 清除 block_hash
```

---

## 2. 推理优化技术

### 2.1 Continuous Batching

**传统批处理**：等待批内所有请求完成才接受新请求 → GPU 利用率低

**连续批处理**：Token 级别调度，短的请求完成后立即腾出空间

```
┌─────────────────────────────────────────────────────┐
│           Continuous Batching 示例                  │
├─────────────────────────────────────────────────────┤
│  Step 1: [Req A] [Req B] [Req C]                  │
│           ↓ ↓ ↓                                  │
│  Step 2: [Req A□] [Req B] [Req C]  (A已完成)    │
│           ↓ ↓ ↓                                  │
│  Step 3:       [Req B□] [Req C] [Req D]         │
│           ↓ ↓ ↓                                  │
│  Step 4: [Req E] [Req C] [Req D]  (B已完成)      │
└─────────────────────────────────────────────────────┘
```

**vLLM V1 调度器**：
- 优先处理 decode 请求（内存带宽密集）
- 混合处理 prefill + decode（V0 不能）
- Chunked Prefill：长prompt分块处理

### 2.2 Prefix Caching

**核心思想**：缓存已计算的 KV 块，新请求相同前缀时复用

**哈希方法**：
```python
# 块哈希计算
block_hash = hash((
    parent_hash,           # 父块哈希
    tuple(block_tokens),   # 块内 token IDs
    extra_hash             # 额外哈希（LoRA ID, 图像hash等）
))
```

**工作流程**：
```
Request 1: "You are a helpful assistant. Translate: Hello"
└── 计算 KV → 缓存 prefix block

Request 2: "You are a helpful assistant. Translate: Goodbye"
└── 查找缓存 → 命中 → 跳过 prefix 计算
```

**性能数据**（Qwen3-32B）：
| 指标 | 无 Prefix Caching | 有 Prefix Caching | 提升 |
|-----|------------------|-------------------|------|
| Output TPS | 427 | 1,513 | +254% |
| Mean TTFT | 4,343ms | 970ms | -78% |

### 2.3 Async Sampling

vLLM 支持多种采样策略的异步执行：
- **Greedy Sampling**
- **Temperature Sampling**
- **Top-K / Top-P Sampling**
- **Guided Decoding**（FSM 约束）

**Speculative Decoding**：
```
Draft: 小模型预测 k 个 token
Verify: 大模型一次验证 k+1 个位置
Accept/Reject: 概率比较接受或拒绝
```

**方法**：
- n-gram：查找历史匹配
- EAGLE：轻量级 MLP draft
- Medusa：多头并行预测

---

## 3. 量化与部署

### 3.1 FP8 量化支持

**KV Cache 量化**：
```bash
# 启用 FP8 KV Cache
vllm serve <model> --kv-cache-dtype fp8
```

**格式选项**：
- `fp8_e4m3`：4 位指数 + 3 位尾数（适合小动态范���）
- `fp8_e5m2`：5 位指数 + 2 位尾数（适合大动态范围）
- `fp8_ds_mla`：MLA 模型专用

**内存节省**：
```
FP16: batch × seq × layers × kv_heads × head_dim × 2 bytes × 2 ≈ 17.2 GB
FP8:  相同公式 × 1 byte ≈ 8.6 GB (50% 节省)
```

**性能数据**：
| 指标 | FP16 KV | FP8 KV | 变化 |
|-----|--------|-------|------|
| Output TPS | 786 | 955 | +22% |
| Max Concurrency | 1.09× | 2.18× | +100% |

### 3.2 AWQ/GPTQ 集成

vLLM 内置支持多种量化：
- **AWQ** (Activation-aware Weight Quantization)
- **GPTQ** (GPT Post-Training Quantization)
- **INT4/INT8**

```python
# 使用量化模型
from vllm import LLM

llm = LLM(
    model="<quantized-model>",
    quantization="awq"  # 或 "gptq"
)
```

### 3.3 多模型部署

**单节点多卡**：
```bash
vllm serve <model> --tensor-parallel-size 4
```

**多节点部署**：
```bash
# Node 1
vllm serve <model> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-start-rank 0 \
  --data-parallel-address <master-ip>

# Node 2
vllm serve <model> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-start-rank 2 \
  --data-parallel-address <master-ip>
```

**Disaggregated P/D**：
```
┌──────────────┐     KV Transfer     ┌──────────────┐
│ Prefill Node │ ──────────────────▶ │  Decode Node │
│   (N GPUs)  │                      │   (M GPUs)   │
└──────────────┘                      └──────────────┘
```

---

## 4. API 服务

### 4.1 OpenAI 兼容 API

```bash
# 启动服务器
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

**端点**：
- `/v1/chat/completions` - Chat API
- `/v1/completions` - Completion API
- `/v1/embeddings` - Embedding API
- `/v1/models` - 模型列表

**调用示例**：
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 50
  }'
```

### 4.2 Gradio/Streamlit 集成

```python
# Gradio Webserver (基于 vLLM)
from gradio.openai_chatbot import create_demo

demo = create_demo(
    title="vLLM Chatbot",
    model="meta-llama/Llama-3.1-8B-Instruct"
)
demo.launch()
```

```python
# Python API
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(
    prompts=["Hello, my name is"],
    sampling_params=SamplingParams(temperature=0.8, top_p=0.95)
)
```

### 4.3 模型多实例管理

**Sleep Mode**：
```bash
# 启用 sleep mode
vllm serve <model> --enable-sleep-mode
```

**睡眠级别**：
- **L1 (Light)**：权重卸载到 CPU，唤醒 2-3 秒
- **L2 (Deep)**：权重丢弃，唤醒 7-8 秒

```bash
# L1 睡眠
curl -X POST 'localhost:8001/sleep?level=1'

# 唤醒
curl -X POST 'localhost:8001/wake_up'

# L2 需要额外重载
curl -X POST 'localhost:8001/collective_rpc' \
  -d '{"method":"reload_weights"}'
```

---

## 5. 源码结构

### 5.1 核心模块解析

```
vllm/
├── v1/                        # V1 引擎
│   ├── core/                  # 核心调度
│   ├── engine/                # 引擎入口
│   ├── executor/              # 执行器
│   ├── worker/                # GPU Worker
│   ├── attention/              # 注意力后端
│   └── sample/                # 采样逻辑
│
├── entrypoints/               # 服务入口
│   ├── openai/               # OpenAI API
│   ├── serve/               # 服务部署
│   └── cli/                  # CLI
│
├── model_executor/            # 模型执行
│   ├── models/               # 模型架构
│   ├── layers/               # 层实现
│   ├── kernels/              # CUDA 内核
│   └── quantization/         # 量化
│
├── distributed/               # 分布式
│   ├── kv_transfer/          # KV 传输
│   └── parallel/              # 并行策略
│
└── inputs/                    # 输入处理
    └── parser/               # Prompt 解析
```

### 5.2 关键类/函数

**LLM 引擎**：
```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    enable_prefix_caching=True
)

# 推理
outputs = llm.generate(prompts, sampling_params)
```

**SamplingParams**：
```python
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    max_tokens=512,
    stop_token_ids=[128001],  # eos token
    include_stop_str_in_result=True
)
```

**Scheduler**：
```python
# 调度策略
- FCFS (First-Come-First-Served)
- Priority (优先级调度)

# 队列管理
- waiting queue: 等待调度的请求
- running queue: 正在执行的请求
```

**KV Cache Manager**：
```python
# 核心方法
- allocate_slots()     # 分配块
- get_computed_blocks() # 获取已计算块
- free()              # 释放块
- find_longest_cache_hit() # 查找缓存命中
```

---

## 6. 性能对比

### 6.1 vLLM vs HuggingFace

| 指标 | HuggingFace | vLLM | 差距 |
|-----|------------|------|-----|
| 吞吐量 | 基准 | 2-4× | vLLM 领先 |
| 内存效率 | 低 | 高 | vLLM 领先 |
| 批处理 | 静态 | 连续 | vLLM 领先 |

**vLLM 优势**：
- PagedAttention 减少内存碎片
- Continuous Batching 提高 GPU 利用率
- 自定义 CUDA 内核优化

### 6.2 vLLM vs llama.cpp

**基准测试**（NVIDIA H200, Llama 3.1 8B）：

| 指标 | vLLM | llama.cpp | 结论 |
|-----|------|-----------|------|
| RPS (64并发) | 35×+ | 基准 | vLLM 领先 |
| TPS (64并发) | 44×+ | 基准 | vLLM 领先 |
| P99 TTFT | 低且稳定 | 指数增长 | vLLM 领先 |
| P99 ITL (低并发) | 较低 | 较高 | vLLM 领先 |

**llama.cpp 优势**：
- 极致便携性（纯 C++，无依赖）
- 快速启动（GGUF 格式）
- CPU/边缘设备支持

**选择建议**：
- **多用户、高吞吐**：vLLM
- **单机、本地开发**：llama.cpp

### 6.3 vLLM vs TGI (Text Generation Inference)

| 特性 | TGI | vLLM |
|-----|-----|------|
| 批处理 | 连续 | 连续 |
| 内存优化 | PagedAttention | PagedAttention |
| 量化 | AWQ/FP8 | AWQ/GPTQ/FP8 |
| API | OpenAI 兼容 | OpenAI 兼容 |
| 多GPU | TP | TP + DP |

**TGI 优势**：
- HuggingFace 生态集成
- 易于部署
- Inferium 生态支持

**vLLM 优势**：
- 更高吞吐量
- 更好的前缀缓存
- 更丰富的优化特性

---

## 参考资料

- [vLLM 官方文档](https://docs.vllm.ai)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06117)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [Inside vLLM - Aleksa Gordic](https://www.aleksagordic.com/blog/vllm)
- [vLLM Optimization - JarvisLabs](https://docs.jarvislabs.ai/blog/vllm-optimization-techniques)