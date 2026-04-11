# 大模型会话与模型权重切换技术调研

> 调研日期: 2026-04-11
> 目标: 系统性调研多模型会话管理、模型权重动态切换技术

---

## 1. 多模型会话管理

### 1.1 会话状态保持

LLM会话状态管理的核心在于保持对话上下文的同时，支持模型的动态切换。主要挑战包括：

#### 关键概念

| 概念 | 描述 |
|------|------|
| **Session ID** | 会话唯一标识符，用于关联上下文 |
| **Context Window** | 模型支持的上下文长度 |
| **KV Cache** | Key-Value缓存，存储注意力计算的中间结果 |
| **System Prompt** | 系统级提示词，定义模型行为 |

#### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Manager                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Session A  │  │ Session B  │  │ Session C  │          │
│  │ - model_id │  │ - model_id │  │ - model_id │          │
│  │ - history │  │ - history │  │ - history │          │
│  │ - kv_cache│  │ - kv_cache│  │ - kv_cache│          │
│  │ - metadata│  │ - metadata│  │ - metadata│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Model Pool                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  │ Model v1 │ │ Model v2 │ │ LoRA A   │                  │
│  └──────────┘ └──────────┘ └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### 代码示例: 会话状态管理

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import time

class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    WARMING = "warming"
    ACTIVE = "active"

@dataclass
class SessionState:
    session_id: str
    user_id: str
    model_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    token_count: int = 0
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.last_active = time.time()

@dataclass  
class ModelInstance:
    model_id: str
    path: str
    status: ModelStatus = ModelStatus.UNLOADED
    ref_count: int = 0
    kv_cache: Optional[Any] = None
    
class SessionManager:
    def __init__(self, max_sessions: int = 1000):
        self.sessions: Dict[str, SessionState] = {}
        self.models: Dict[str, ModelInstance] = {}
        self.max_sessions = max_sessions
        
    def create_session(self, user_id: str, model_id: str, 
                     system_prompt: str = "") -> str:
        """创建新会话"""
        import uuid
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            user_id=user_id,
            model_id=model_id,
            system_prompt=system_prompt
        )
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """获取会话状态"""
        return self.sessions.get(session_id)
    
    def switch_model(self, session_id: str, new_model_id: str) -> bool:
        """切换会话模型 - 需要处理上下文迁移"""
        session = self.sessions.get(session_id)
        if not session:
            return False
            
        old_model_id = session.model_id
        session.model_id = new_model_id
        session.last_active = time.time()
        
        # 注意: 切换模型会清除历史，因为不同模型有不同tokenizer
        session.history.clear()
        return True
```

### 1.2 模型上下文隔离

多租户场景下的上下文隔离是核心安全需求：

```
┌────────────────────────────────────────────────────────────┐
│                    Multi-Tenant Isolation                │
├────────────────────────────────────────────────────────────┤
│                                                        │
│  Tenant A          Tenant B          Tenant C             │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │ Context │      │ Context │      │ Context │          │
│  │ [0,1,2] │      │ [0,1,2] │      │ [0,1,2] │          │
│  └──────────┘      └──────────┘      └──────────┘          │
│        ↓                  ↓                  ↓                  │
│  ┌──────────────────────────────────────────┐          │
│  │        KV Cache (隔离存储)                   │          │
│  │  Tenant A: [████████████████████████]        │          │
│  │  Tenant B: [████████████████████████]    │          │
│  │  Tenant C: [████████████████████████]    │          │
│  └──────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

#### 隔离策略

1. **进程级隔离**: 每个租户独立进程
2. **内存级隔离**: GPU内存分區隔离
3. **KV Cache隔离**: 防止cache泄露

### 1.3 会话恢复与迁移

```
┌─────────────────────────────────────────────────────────────┐
│               Session Recovery Flow                      │
├───────────────────────────────────────────────────���─��───────┤
│                                                      │
│  Client          API Gateway      Session Store          │
│    │                 │               │                   │
│    ├─ POST /chat ───>│               │                   │
│    │                 ├── 查找会话 ──>│                   │
│    │                 │<─ 返回状态 ───┤                   │
│    │                 │               │                   │
│    │<─ 恢复对话 ────┤               │                   │
│                                                      │
└─────────────────────────────────────────────────────────────┘
```

```python
import json
import redis

class SessionStore:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    def save_session(self, session: SessionState, ttl: int = 86400):
        """持久化会话状态"""
        key = f"session:{session.session_id}"
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "model_id": session.model_id,
            "history": session.history,
            "system_prompt": session.system_prompt,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "token_count": session.token_count
        }
        self.redis.setex(key, ttl, json.dumps(data))
        
    def load_session(self, session_id: str) -> Optional[SessionState]:
        """恢复会话"""
        key = f"session:{session_id}"
        data = self.redis.get(key)
        if not data:
            return None
            
        obj = json.loads(data)
        session = SessionState(**obj)
        return session
```

---

## 2. 模型权重动态切换

### 2.1 LoRA适配器切换

LoRA (Low-Rank Adaptation) 是最常用的轻量级微调方法，可以热切换：

#### 架构

```
┌─────────────────────────────────────────────────────────────┐
│              LoRA Adapter Switching                       │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  Base Model (冻结)                                      │
│  ┌─────────────────────────────────────────┐           │
│  │  ┌─────┐   ┌─────┐   ┌─────┐            │           │
│  │  │ Q  │   │ K  │   │ V  │            │           │
│  │  └──┬──┘   └──┬──┘   └──┬──┘            │           │
│  │     │         │         │                │           │
│  │   ΔW        ΔW        ΔW   (可学习)  │           │
│  └─────┴─────────┴─────────┴────────────┘           │
│                                                      │
│  LoRA A        LoRA B        LoRA C                    │
│  (数学)       (代码)       (对话)                      │
│                                                      │
└─────────────────────────────────────────────────────���─���─────┘
```

#### vLLM LoRA服务

```python
# vLLM 使用 LoRA adapter示例
from vllm import LLM, SamplingParams

# 初始化带LoRA支持的引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_lora_rank=64,
    max_num_seqs=256,
)

# 定义多个adapter
adapters = {
    "math": "/path/to/math-adapter",
    "code": "/path/to/code-adapter", 
    "chat": "/path/to/chat-adapter"
}

def generate_with_lora(prompt: str, adapter_name: str) -> str:
    """使用指定LoRA adapter生成"""
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        lora_name=adapter_name  # 指定adapter
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# 批量请求 - 不同adapter在同一批次处理
def batch_generate(prompts: list, adapter_names: list) -> list:
    """批量生成，自动路由到对应adapter"""
    sampling_params = [
        SamplingParams(
            temperature=0.7,
            max_tokens=512,
            lora_name=name
        ) for name in adapter_names
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]
```

#### LoRA权重管理

```python
class LoRAManager:
    def __init__(self, base_model: str, cache_dir: str):
        self.base_model = base_model
        self.cache_dir = cache_dir
        self.loaded_adapters: Dict[str, object] = {}
        self.lru_cache = OrderedDict()  # LRU缓存
        
    def load_adapter(self, adapter_id: str, adapter_path: str):
        """加载LoRA adapter"""
        from peft import PeftModel, load_adapter
        
        if adapter_id in self.loaded_adapters:
            # 提升缓存命中率
            self.lru_cache.move_to_end(adapter_id)
            return
            
        # 检查缓存上限
        if len(self.loaded_adapters) >= self.max_adapters:
            self._evict_lru()
            
        adapter = load_adapter(adapter_path)
        self.loaded_adapters[adapter_id] = adapter
        self.lru_cache[adapter_id] = True
        
    def _evict_lru(self):
        """驱逐最久未使用的adapter"""
        oldest = next(iter(self.lru_cache))
        del self.loaded_adapters[oldest]
        del self.lru_cache[oldest]
        
    def switch_adapter(self, adapter_id: str) -> bool:
        """热切换adapter"""
        if adapter_id not in self.loaded_adapters:
            return False
        self.lru_cache.move_to_end(adapter_id)
        return True
```

### 2.2 量化权重切换

不同量化级别的权重切换：

#### 量化方案对比

| 量化方法 | 精度 | 显存减少 | 性能影响 |
|---------|------|---------|---------|
| FP16 | 16-bit | 1x | 无 |
| INT8 | 8-bit | ~50% | ~2% |
| INT4 | 4-bit | ~75% | ~5-10% |
| GPTQ | 4-bit | ~75% | ~3-5% |
| AWQ | 4-bit | ~75% | ~2% |

#### 动态量化切换

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantizedModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.models: Dict[str, object] = {}
        
    def load_quantized(self, quantization: str = "int4"):
        """加载量化模型"""
        loading_mapping = {
            "fp16": self._load_fp16,
            "int8": self._load_int8,
            "int4": self._load_int4
        }
        
        if quantization not in loading_mapping:
            raise ValueError(f"不支持的量化方法: {quantization}")
            
        model, tokenizer = loading_mapping[quantization]()
        self.models[quantization] = {"model": model, "tokenizer": tokenizer}
        return model, tokenizer
        
    def _load_fp16(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer
        
    def _load_int8(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer
        
    def _load_int4(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer
        
    def switch_quantization(self, from_q: str, to_q: str) -> bool:
        """切换量化级别"""
        if to_q not in self.models:
            self.load_quantized(to_q)
            
        # 交换模型引用
        self.models[from_q], self.models[to_q] = \
            self.models[to_q], self.models[from_q]
        return True
```

### 2.3 动态加载 vs 热替换

| 方案 | 加载时间 | 显存 | 适用场景 |
|------|---------|------|---------|
| 动态加载 | ~10-30s | 共享 | 小团队、多模型 |
| 热替换 | ~1-3s | 独占 | 高并发、生产环境 |
| 多实例 | ~0s | 并列 | 大规模服务 |

```python
class ModelSwitchStrategy(Enum):
    DYNAMIC = "dynamic"      # 动态加载
    HOT_SWAP = "hot_swap"    # 热替换
    MULTI_INSTANCE = "multi" # 多实例

class ModelSwitcher:
    def __init__(self, strategy: ModelSwitchStrategy):
        self.strategy = strategy
        self.current_model = None
        
    def switch(self, target_model: str) -> float:
        """切换模型并返回切换延迟"""
        import time
        start = time.time()
        
        if self.strategy == ModelSwitchStrategy.DYNAMIC:
            self._dynamic_load(target_model)
        elif self.strategy == ModelSwitchStrategy.HOT_SWAP:
            self._hot_swap(target_model)
        else:
            self._select_instance(target_model)
            
        return time.time() - start
        
    def _dynamic_load(self, model_id: str):
        """动态加载 - 卸载当前，加载新模型"""
        if self.current_model:
            self._unload_model(self.current_model)
        self.current_model = self._load_model(model_id)
        
    def _hot_swap(self, model_id: str):
        """热替换 - 使用预加载的模型"""
        if model_id not in self.preloaded_models:
            # 后台预加载
            self.preload_async(model_id)
            
        self.current_model = self.preloaded_models[model_id]
        
    def _select_instance(self, model_id: str):
        """实例选择 - 直接切换实例"""
        self.current_model = self.instances[model_id]
```

---

## 3. 主流方案

### 3.1 vLLM动态批处理

vLLM是最流行的开源推理服务框架，支持动态LoRA和批处理：

#### 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │              API Server (FastAPI)                 │    │
│  │  - /v1/chat/completions                         │    │
│  │  - /v1/completions                             │    │
│  └──────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │           Scheduler + Batch Dispatcher             │    │
│  │  - PagedAttention                              │    │
│  │  - KV Cache Management                         │    │
│  │  - Multi-LoRA Serving                         │    │
│  └──────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │              GPU Worker (CUDA)                   │    │
│  │  - Continuous Batching                        │    │
│  │  - Speculative Decoding                       │    │
│  │  - LoRA Kernel                              │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

#### ��码��例

```python
# vLLM 服务启动
# 命令行
"""
vllm serve meta-llama/Llama-2-7b-hf \
    --dtype half \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 64 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9
"""

# Python API
from vllm import LLM, SamplingParams

# 初始化引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # 多卡并行
    dtype="half",
    max_num_seqs=256,
    gpu_memory_utilization=0.9,
)

# 单次请求
def chat(prompt: str,model_id: str, system_prompt: str = None) -> str:
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        stop=["</s>"],
    )
    
    # 构建消息格式
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # 使用chatglm3 streamchat
    from vllm.utils import get_chat_template
    prompt = get_chat_template(messages, tokenizer=llm.get_tokenizer())
    
    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text

# 批量推理
def batch_chat(requests: list) -> list:
    """批量处理，自动调度"""
    prompts = [r["prompt"] for r in requests]
    params = [
        SamplingParams(
            temperature=r.get("temperature", 0.7),
            max_tokens=r.get("max_tokens", 512),
        ) for r in requests
    ]
    
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]
```

### 3.2 Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load(self, device: str = "cuda", dtype: str = "float16"):
        """加载模型"""
        torch_dtype = getattr(torch, dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )
            
        return self.tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
    def reload(self, new_model_path: str):
        """重新加载模型"""
        self.model_path = new_model_path
        self.load()
```

### 3.3 llama.cpp 权重热加载

llama.cpp 支持Router模式，可管理多个模型：

#### 架构

```
┌─────────────────────────────────────────────────────────────┐
│           llama.cpp Server (Router Mode)                     │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  Server Process                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Model Registry                                 │  │
│  │  - model_1.gguf (loaded)                       │  │
│  │  - model_2.gguf (ready)                       │  │
│  │  - model_3.gguf (ready)                       │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Request Router                               │  │
│  │  /v1/chat/completions/{model}                  │  │
│  │  /v1/completions/{model}                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 启动服务

```bash
# 启动llama.cpp server (Router模式)
llama-server \
    --model ./model1.gguf \
    --port 8080 \
    --hostname 0.0.0.0

# 添加更多模型 (通过API)
# POST /v1/model/add
curl -X POST http://localhost:8080/v1/model/add \
    -H "Content-Type: application/json" \
    -d '{
        "name": "model2",
        "filename": "./model2.gguf"
    }'

# 切换模型
# POST /v1/model/set
curl -X POST http://localhost:8080/v1/model/set \
    -d '{"model": "model2"}'
```

#### Python API

```python
import llama

class LlamaCppManager:
    def __init__(self, model_path: str, n_ctx: int = 4096):
        self.model_path = model_path
        self.params = llama.LlamaModelParams()
        self.params.n_ctx = n_ctx
        
    def load_model(self):
        """加载模型"""
        self.model = llama.load_model_from_file(
            self.model_path,
            self.params
        )
        self.ctx = llama.NewContext(self.model, self.params)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成"""
        ctxt = self.ctx
        max_tokens = kwargs.get("max_tokens", 512)
        temp = kwargs.get("temperature", 0.7)
        
        # 创建采样器
        sampler = llama.Sampler([{
            "id": "temperature",
            "temp": temp
        }])
        
        # 生成
        results = ctxt.completions(prompt, sampler, max_tokens)
        return results[0].text
        
    def switch_model(self, new_model_path: str):
        """切换模型"""
        self.model = llama.load_model_from_file(
            new_model_path,
            self.params
        )
        self.ctx = llama.NewContext(self.model, self.params)
```

### 3.4 模型多实例管理

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Instance Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Instance 1│  │ Instance 2│  │ Instance 3│       │
│  │ (Qwen)    │  │ (DeepSeek)│  │ (Llama)   │       │
│  │ GPU 0     │  │ GPU 1     │  │ GPU 2     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│         ↓                ↓                ↓              │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Load Balancer                        │  │
│  │  - Round Robin                                   │  │
│  │  - Least Connections                             │  │
│  │  - Weighted Response Time                       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

```python
import asyncio
from multiprocessing import Process, Queue

class ModelPool:
    def __init__(self, model_configs: list, replicas_per_model: int = 1):
        self.pools: Dict[str, list] = {}
        self.replicas = replicas_per_model
        
        for config in model_configs:
            self.pools[config["model_id"]] = []
            for i in range(replicas_per_model):
                instance = ModelInstance(
                    model_id=config["model_id"],
                    model_path=config["path"],
                    gpu_id=config.get("gpu_ids", [0])[i % len(config["gpu_ids"])]
                )
                self.pools[config["model_id"]].append(instance)
                
    def get_instance(self, model_id: str) -> 'ModelInstance':
        """获取可用实例"""
        pool = self.pools.get(model_id, [])
        # 简单的负载均衡
        return min(pool, key=lambda x: x.busy_count)
        
class ModelInstance:
    def __init__(self, model_id: str, model_path: str, gpu_id: int):
        self.model_id = model_id
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.busy_count = 0
        self.process = None
        
    def start(self):
        """启动实例进程"""
        self.process = Process(
            target=self._run_model,
            args=(self.model_path, self.gpu_id)
        )
        self.process.start()
        
    def _run_model(self, model_path: str, gpu_id: int):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 启动模型服务
        # ...
```

---

## 4. 实现方案

### 4.1 内存管理

```python
import gc
import torch

class MemoryManager:
    def __init__(self, gpu_memory_limit: float = 0.9):
        self.gpu_memory_limit = gpu_memory_limit
        self.current_models: Dict[str, object] = {}
        
    def get_gpu_memory(self) -> dict:
        """获取GPU显存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "allocated": allocated,
                "reserved": reserved,
                "total": total,
                "free": total * self.gpu_memory_limit - allocated
            }
        return {}
        
    def can_load(self, model_size: float) -> bool:
        """检查是否可以加载模型"""
        mem = self.get_gpu_memory()
        return mem.get("free", 0) > model_size
        
    def unload_model(self, model_id: str):
        """卸载模型"""
        if model_id in self.current_models:
            del self.current_models[model_id]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def evict_lru(self):
        """LRU驱逐"""
        # 按最后使用时间排序
        sorted_models = sorted(
            self.current_models.items(),
            key=lambda x: x[1].last_used
        )
        if sorted_models:
            self.unload_model(sorted_models[0][0])
```

### 4.2 KV Cache复用

```
┌─────────────────────────────────────────────────────────────┐
│            KV Cache Sharing Strategy                     │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  原始: 每个请求独立计算                                │
│  ┌─────┐      ┌─────┐      ┌─────┐                 │
│  │ Req │      │ Req │      │ Req │                 │
│  │  A  │      │  B  │      │  C  │                 │
│  └──┬──┘      └──┬──┘      └──┬──┘                 │
│     ↓             ↓             ↓                     │
│  [KV A]        [KV B]        [KV C]                 │
│                                                      │
│  优化: 相同前缀共享KV Cache                           │
│                                                      │
│  System: "You are a helpful assistant"               │
│  └──────────────────────────────────────┐           │
│                                         ↓           │
│  ┌─────────────────────────────────────────────┐    │
│  │     Shared KV [System Prompt]                     │    │
│  └─────────────────────────────────────────────┘    │
│                        ↓                              │
│  ┌─────┐      ┌─────┐      ┌─────┐                 │
│  │Unique│      │Unique│      │Unique│                 │
│  │ KV  │      │ KV  │      │ KV  │                 │
│  └─────┘      └─────┘      └─────┘                 │
└─────────────────────────────────────────────────────────────┘
```

```python
class KVCacheManager:
    def __init__(self, max_cache_size: int = 1000):
        self.cache: Dict[str, object] = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        
    def get_cache_key(self, prompt: str, model_id: str) -> str:
        """生成cache key"""
        import hashlib
        prefix = prompt[:100]  # 使用前100 tokens
        return hashlib.md5(
            f"{model_id}:{prefix}".encode()
        ).hexdigest()
        
    def get_or_compute(self, prompt: str, model_id: str, 
                     compute_fn) -> object:
        """获取或计算KV Cache"""
        key = self.get_cache_key(prompt, model_id)
        
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
            
        # 计算
        kv_cache = compute_fn(prompt)
        
        # 加入缓存
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
            
        self.cache[key] = kv_cache
        return kv_cache
        
    def share_across_requests(self, prompts: list) -> Dict[str, object]:
        """跨请求共享KV"""
        # 寻找共同前缀
        common_prefix = self._find_common_prefix(prompts)
        
        if common_prefix:
            # 先计算共同前缀的KV
            prefix_kv = self._compute_prefix_kv(common_prefix)
            return {
                "prefix": prefix_kv,
                "unique": {}
            }
        return {}
        
    def _find_common_prefix(self, prompts: list) -> str:
        """找共同前缀"""
        if not prompts:
            return ""
            
        words = prompts[0].split()
        common = words.copy()
        
        for p in prompts[1:]:
            p_words = p.split()
            common = [w for w in common if w in p_words]
            
        return " ".join(common)
```

### 4.3 模型预热

```python
class ModelWarmer:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    def warm_model(self, model_id: str, warmup_prompts: list = None):
        """预热模型"""
        if warmup_prompts is None:
            warmup_prompts = [
                "Hello, how are you?",
                "What is 1+1?",
                "Explain quantum computing.",
            ]
            
        model = self.model_manager.get_model(model_id)
        
        for prompt in warmup_prompts:
            model.generate(prompt, max_tokens=1)
            
        # 第一次调用较慢，后续会使用cache
        return True
        
    def background_warm(self, model_id: str):
        """后台预热"""
        import threading
        
        def warm():
            self.warm_model(model_id)
            
        thread = threading.Thread(target=warm, daemon=True)
        thread.start()
        return thread
        
    def preload_after_unused(self, model_id: str, timeout: int = 300):
        """闲置超时后预加载"""
        import time
        import threading
        
        def delayed_warm():
            time.sleep(timeout)
            self.warm_model(model_id)
            
        thread = threading.Thread(target=delayed_warm, daemon=True)
        thread.start()
```

---

## 5. 推荐方案

### 方案对比

| 方案 | 易用性 | 性能 | 灵活性 | 适用场景 |
|------|--------|------|--------|---------|
| vLLM | ★★★★★ | ★★★★★ | ★★★★☆ | 生产环境首选 |
| HF Transformers | ★★★★☆ | ★★★★☆ | ★★★★★ | 研究和原型 |
| llama.cpp | ★★★★☆ | ★★★★☆ | ★★★☆☆ | 本地部署 |
| TGI (HuggingFace) | ★★★★★ | ★★★★★ | ★★★☆☆ | 云端部署 |

### 推荐架构

```
┌─────────────────────────────────────────────────────────────┐
│               Recommended Architecture                       │
├───────────────────────────────────────────────────────���─���───┤
│                                                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │            API Gateway (FastAPI)                  │    │
│  │  - 认证/授权                                    │    │
│  │  - 请求路由                                      │    │
│  │  - 限流                                        │    │
│  └──────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Session Manager (Redis + State)            │    │
│  │  - 会话状态存储                                  │    │
│  │  - 模型路由                                      │    │
│  │  - KV Cache管理                                 │    │
│  └──────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │         Model Pool (vLLM Engine)                │    │
│  │  - 多模型管理                                    │    │
│  │  - LoRA动态切换                                 │    │
│  │  - 动态批处理                                  │    │
│  └──────────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────────────┘
```

### 快速开始配置

```yaml
# docker-compose.yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - MODEL=meta-llama/Llama-2-7b-hf
      - GPU_MEMORY_UTILIZATION=0.9
      - MAX_NUM_SEQS=256
      - ENABLE_LORA=true
    volumes:
      - ./models:/models

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 参考资源

- [vLLM Documentation](https://docs.vllm.ai/)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [KVShare Research](https://arxiv.org/abs/2503.16525)