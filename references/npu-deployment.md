# NPU 部署量化模型完整技术方案

> 本文档涵盖 NPU 部署量化模型的核心技术方案，包括架构、量化方法、图优化、编译流程和部署流程。

## 📋 目录

1. [NPU 部署整体架构](#1-npu-部署整体架构)
2. [模型量化到 NPU](#2-模型量化到-npu)
3. [计算图优化](#3-计算图优化)
4. [NPU 编译流程](#4-npu-编译流程)
5. [部署流程](#5-部署流程)

---

## 1. NPU 部署整体架构

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    推理引擎 (Inference Engine)                 │
├──────────────────────┬──────────────────────┬───────────────────┤
│   模型优化器           │    运行时             │   调度器           │
│   (Model Optimizer)   │    (Runtime)         │   (Scheduler)      │
├──────────────────────┴──────────────────────┴───────────────────┤
│                    编译器 (Compiler)                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ 前端        │→ │ 中间表示     │→ │ 后端        │        │
│   │ (Frontend) │  │   (IR)      │  │ (Codegen)  │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                  硬件抽象层 (HAL - Hardware Abstraction Layer)            │
├──────────────────────┬──────────────────────┬───────────────────┤
│   计算单元             │   内存管理             │   I/O 管理         │
│   (Compute Units)     │   (Memory Mgmt)       │   (I/O Mgmt)       │
├──────────────────────┴──────────────────────┴───────────────────┤
│                     NPU 硬件 (NPU Hardware)                           │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐   │
│   │ Tensor Core │  │ Vector Core │  │ Scalar Core │  │ Cache     │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘   │
└────────────────────────────────────────────────────────��────────────────┘
```

### 1.2 硬件抽象层 (HAL)

硬件抽象层提供统一的硬件访问接口，屏蔽底层硬件差异：

```cpp
// HAL 示例接口
class NPUHal {
public:
    // 内存分配与释放
    virtual void* allocateBuffer(size_t size, MemoryType type) = 0;
    virtual void freeBuffer(void* ptr) = 0;
    
    // 计算核调度
    virtual void launchKernel(const Kernel& kernel, const LaunchConfig& config) = 0;
    
    // 数据传输
    virtual void copyData(void* dst, const void* src, size_t size, Direction dir) = 0;
    
    // 同步
    virtual void synchronize() = 0;
};
```

### 1.3 运行时 (Runtime)

运行时负责模型的生命周期管理和执行调度：

```python
# 运行时伪代码
class NPURuntime:
    def __init__(self, model_path, num_threads=4):
        self.model = self._load_model(model_path)
        self.executor = NPUExecutor(self.model)
        
    def _load_model(self, model_path):
        """加载编译好的模型"""
        with open(model_path, 'rb') as f:
            return self._deserialize(f.read())
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理"""
        # 创建输入tensor
        input_tensors = self._create_tensors(inputs)
        
        # 执行推理
        output_tensors = self.executor.execute(input_tensors)
        
        # 转换为numpy
        return self._to_numpy(output_tensors)
    
    def __del__(self):
        self.executor.release()
```

### 1.4 编译器流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ PyTorch │ or │  ONNX   │ or │ TensorFlow│ or│ 其他框架 │
│  Model  │    │ Model   │    │  Model   │    │  Model  │
└────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
     │               │               │               │
     └───────┬───────┴───────┬───────┴───────┬───────┘
             │               │               │
             ▼               ▼               ▼
     ┌──────────────────────────────────────────────┐
     │           前端 (Frontend - 模型解析)              │
     │   • PyTorch→ONNX 转换                          │
     │   • ONNX 解析器                               │
     │   • 算子图构建                               │
     └────────────────────┬───────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────┐
     │         中间表示 (IR - 計算图优化)               │
     │   • 算子融合                                 │
     │   • 布局转换                                 │
     │   • 内存优化                                 │
     │   • 量化感知优化                             │
     └────────────────────┬───────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────┐
     │           后端 (Backend - 代码生成)             │
     │   • 目��代码生成                             │
     │   • 内存分配                               │
     │   • 可执行文件打包                          │
     └────────────────────┬───────────────────────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ NPU 可执行  │
                  │    文件    │
                  └──────────────┘
```

---

## 2. 模型量化到 NPU

### 2.1 量化概述

量化是将浮点模型转换为低精度整数表示的技术，核心公式：

$$X_{int8} = round(\frac{X_{fp32} - Z}{S})$$

$$X_{fp32} = X_{int8} \times S + Z$$

其中：
- $S$：缩放因子 (Scale)
- $Z$：零点 (Zero Point)

### 2.2 量化感知训练 (QAT)

量化感知训练在训练阶段模拟量化效果，使模型适应低精度推理：

```python
# 使用 TorchAO 进行 QAT
import torch
from torchao.quantization import quantize_, Int8DynamicActivationIntxWeightConfig, PerGroup
from torchao.quantization.qat import QATConfig

# 准备 QAT 配置
base_config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),
)

# 步骤1: 准备阶段 - 插入伪量化节点
quantize_(my_model, QATConfig(base_config, step="prepare"))

# 步骤2: 训练模型 (带量化模拟)
# ... 训练代码 ...

# 步骤3: 转换阶段 - 转换为真正的量化模型
quantize_(my_model, QATConfig(base_config, step="convert"))

# 保存量化模型
torch.save(my_model.state_dict(), "quantized_model.pth")
```

**QAT 优势**：
- 恢复 96% 量化精度损失
- 在低比特 (int4) 场景效果显著

### 2.3 训练后量化 (PTQ)

PTQ 在训练后进行量化，更快速但可能有精度损失：

```python
# 使用 TorchAO 进行 PTQ - int4 权重量化
import torch
from torchao.quantization import (
    Int4WeightOnlyConfig, 
    quantize_
)

# CUDA 量化
if torch.cuda.is_available():
    quantize_(
        model, 
        Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"
        )
    )
# XPU 量化
elif torch.xpu.is_available():
    quantize_(
        model, 
        Int4WeightOnlyConfig(
            group_size=32,
            int4_packing_format="plain_int32"
        )
    )
```

### 2.4 INT8 量化方案

```python
# INT8 动态激活量化
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow
)

# 创建量化配置
quantization_config = TorchAoConfig(
    quant_type=Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow()
    )
)

# 加载并自动量化
quantized_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)
```

### 2.5 INT4 量化方案

```python
# INT4 权重量化 (W4A16)
from torchao.quantization import (
    Int4WeightOnlyConfig,
    quantize_
)

# group_size=32 表示每32个权重共享一个量化参数
quantize_(
    model,
    Int4WeightOnlyConfig(
        group_size=32,
        int4_packing_format="tile_packed_to_4d"
    )
)
```

### 2.6 混合精度量化

混合精度在不同层使用不同量化策略：

```python
# 混合精度配置示例
mixed_precision_config = {
    # 核心层使用较高精度
    "attention.qkv": {"weight": "int8", "activation": "fp16"},
    "attention.dense": {"weight": "int8", "activation": "fp16"},
    # FFN 层可���使���更低精度
    "mlp.gate_proj": {"weight": "int4", "activation": "int8"},
    "mlp.up_proj": {"weight": "int4", "activation": "int8"},
    "mlp.down_proj": {"weight": "int8", "activation": "fp16"},
}

# 应用混合精度
for layer_name, config in mixed_precision_config.items():
    apply_quantization(model, layer_name, config)
```

---

## 3. 计算图优化

### 3.1 图融合 (Kernel Fusion)

图融合将多个算子合并为一个，减少内存访问和内核启动开销：

```
融合前：                           融合后：
┌─────────┐    ┌─────────┐         ┌─────────────┐
│  Conv   │→  │  ReLU  │         │ Conv+ReLU │
└─────────┘    └─────────┘         └───────────┘
                   
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Conv   │→  │  BN    │→  │  ReLU  │    →   ┌─────────────┐
└─────────┘    └─────────┘    └─────────┘      │ Conv+BN+ReLU│
                                                 └───────────┘
```

```python
# 图融合示例 (伪代码)
def apply_kernel_fusion(graph):
    """应用算子融合"""
    fusion_rules = [
        # 标准融合模式
        ["Conv2d", "BatchNorm2d", "ReLU"] -> "FusedConvBNReLU",
        ["Conv2d", "ReLU"] -> "FusedConvReLU",
        ["Linear", "ReLU"] -> "FusedLinearReLU",
        ["MatMul", "Add"] -> "FusedMatMulAdd",
    ]
    
    for pattern, fused_op in fusion_rules:
        if graph.match_pattern(pattern):
            graph.fuse_to(fused_op)
    
    return graph
```

### 3.2 算子融合规则

| 融合模式 | 融合后算子 | 性能提��� |
|---------|-----------|---------|
| Conv + BN + ReLU | FusedConvBNReLU | 减少 2 次内存读写 |
| Conv + ReLU | FusedConvReLU | 减少 1 次内存读写 |
| Linear + ReLU | FusedLinearReLU | 减少 1 次内存读写 |
| MatMul + Add (Bias) | FusedMatMulBias | 融合为单内核 |
| Softmax + Reshape | FusedSoftmax | 减少内存分配 |

### 3.3 内存优化

```python
# 内存优化策略
class MemoryOptimizer:
    def __init__(self, graph):
        self.graph = graph
        
    def optimize(self):
        # 1. 内存池管理
        self._enable_memory_pool()
        
        # 2. 原地操作
        self._enable_inplace_ops()
        
        # 3. 显存复用
        self._enable_memory_reuse()
        
        # 4. 内存分配器优化
        self._optimizeAllocator()
        
        return self.graph
    
    def _enable_inplace_ops(self):
        """启用原地操作"""
        inplace_ops = [
            "relu", "sigmoid", "tanh",
            "clip", "resize", "reshape"
        ]
        for op in inplace_ops:
            self.graph.set_inplace(op, True)
```

### 3.4 布局转换 (Layout Transform)

布局转换优化张量内存排布以匹配 NPU 计算特性：

```python
# 布局转换示例
class LayoutTransformer:
    # NPU 常用布局
    LAYOUT_NCHW = "NCHW"      # 标准通道优先
    LAYOUT_NHWC = "NHWC"       # 空间优先 (NPU 优化)
    LAYOUT_CHWN = "CHWN"        # 通道-高度-宽度-批量
    LAYOUT_BLOCKED = "BLOCKED"   # 分块布局
    
    def transform(self, tensor, target_layout):
        """执行布局转换"""
        if tensor.layout == target_layout:
            return tensor
            
        # 检测是否需要转换
        current_layout = tensor.layout
        
        # 优化转换策略
        if self._should_fuse_transform(current_layout, target_layout):
            # 融合到相邻算子
            return self._fuse_transform(tensor, target_layout)
        else:
            # 独立转换
            return self._insert_transform_op(tensor, target_layout)
```

---

## 4. NPU 编译流程

### 4.1 前端：模型解析 (PyTorch/ONNX)

```python
# PyTorch 模型转换为 ONNX
import torch
import torch.onnx

def convert_pytorch_to_onnx(pytorch_model, input_sample, output_path):
    """PyTorch → ONNX 转换"""
    pytorch_model.eval()
    
    torch.onnx.export(
        pytorch_model,
        input_sample,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
```

### 4.2 中间表示 (IR)

```
ONNX IR 结构:
┌────────────────────────────────────────────┐
│              ModelProto                     │
│  ┌──────────────────────────────────────┐ │
│  │           GraphProto                  │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │ node: List[NodeProto]            │ │ │
│  │  │   - input[]                     │ │ │
│  │  │   - output                      │ │ │
│  │  │   - op_type                     │ │ │
│  │  │   - attribute[]                  │ │ │
│  │  └─────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │ initializer: List[TensorProto]   │ │ │
│  │  │   - name, dims, data_type        │ │ │
│  │  │   - raw_data (权重)             │ │ │
│  │  └─────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │ value_info: List[TensorProto]     │ │ │
│  │  │   - 中间张量类型信息             │ │ │
│  │  └─────────────────────────────────┘ │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

### 4.3 后端：代码生成

```python
# 代码生成流程
class NPUCodegen:
    def __init__(self, ir_graph, target_npu):
        self.ir = ir_graph
        self.target = target_npu
        
    def generate(self):
        # 1. 调度生成
        schedule = self._generate_schedule()
        
        # 2. 代码生成
        code = self._generate_code(schedule)
        
        # 3. 链接信息生成
        link_info = self._generate_link_info()
        
        # 4. 打包
        return self._package(code, link_info)
    
    def _generate_schedule(self):
        """生成计算调度"""
        # 拓扑排序
        nodes = self.ir.topological_sort()
        
        # 调度优化
        schedule = []
        for node in nodes:
            if self._can_fuse(node):
                schedule.append(self._create_fused_node(node))
            else:
                schedule.append(node)
                
        return schedule
```

### 4.4 内存分配

```python
# 内存分配策略
class MemoryAllocator:
    def __init__(self, ir_graph):
        self.ir = ir_graph
        self.buffer_pool = {}
        
    def allocate(self):
        # 1. 分析数据依赖
        deps = self._analyze_dependencies()
        
        # 2. 计算生命周期
        lifetimes = self._compute_lifetimes()
        
        # 3. 分配内存块
        allocation = {}
        for tensor, lifetime in lifetimes.items():
            if self._can_reuse(tensor, allocation):
                tensor.buffer = self._find_reusable_buffer(tensor)
            else:
                tensor.buffer = self._allocate_new(tensor.size)
                
        return allocation
    
    def _analyze_dependencies(self):
        """分析数据依赖关系"""
        # ... 依赖分析代码 ...
        pass
```

---

## 5. 部署流程

### 5.1 模型转换

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ PyTorch     │     │   ONNX      │     │   中间表示  │
│ (FP32)      │ →   │   Model    │ →   │   (IR)     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                      │
       │ 量化                              优化
       ▼                                      ▼
┌─────────────┐                       ┌─────────────┐
│ PyTorch     │                       │ 优化后的 IR │
│ (INT8/INT4)│                       │           │
└─────────────┘                       └──────��─��────┘
```

```python
# 完整模型转换流程
def convert_model_to_npu(
    model,
    input_shape,
    output_path,
    quantization_config=None
):
    # 1. 转换 PyTorch → ONNX
    dummy_input = torch.randn(*input_shape)
    onnx_path = "/tmp/model.onnx"
    convert_pytorch_to_onnx(model, dummy_input, onnx_path)
    
    # 2. 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    
    # 3. 应用量化 (如果配置)
    if quantization_config:
        onnx_model = apply_quantization(onnx_model, quantization_config)
    
    # 4. 编译为 NPU 模型
    npu_model = compile_to_npu(onnx_model)
    
    # 5. 保存编译后的模型
    with open(output_path, 'wb') as f:
        npu_model.serialize(f)
    
    return output_path
```

### 5.2 量化校准

```python
# 量化校准流程
class QuantizationCalibrator:
    def __init__(self, model, calibration_data):
        self.model = model
        self.data = calibration_data
        
    def calibrate(self):
        # 收集激活值统计
        activation_stats = {}
        
        for i, batch in enumerate(self.data):
            # 运行推理收集统计
            outputs = self.model(batch)
            
            # 收集每层输出统计
            for name, tensor in outputs.items():
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(self._compute_stats(tensor))
                
            if i >= 100:  # 校准数据量
                break
        
        # 计算量化参数
        quant_params = {}
        for name, stats in activation_stats.items():
            quant_params[name] = self._compute_scale_zp(stats)
            
        return quant_params
    
    def _compute_stats(self, tensor):
        """计算张量统计"""
        return {
            'min': tensor.min(),
            'max': tensor.max(),
            'mean': tensor.mean(),
            'std': tensor.std()
        }
    
    def _compute_scale_zp(self, stats_list):
        """计算量化缩放因子和零点"""
        min_val = min(s['min'] for s in stats_list)
        max_val = max(s['max'] for s in stats_list)
        
        scale = (max_val - min_val) / 255.0
        zero_point = -min_val / scale
        
        return {'scale': scale, 'zero_point': zero_point}
```

### 5.3 编译部署

```bash
# NPU 模型编译命令示例
npu编译器 \
    --input model.onnx \
    --output model.npu \
    --quantization int8 \
    --calibration-data calibration.npz \
    --optimization-level O3 \
    --target-npu npu8800 \
    --memory-budget 512MB
```

```python
# 部署示例
import numpy as np

def deploy_model(model_path):
    # 1. 加载编译好的模型
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    # 2. 初始化 NPU 运行时
    runtime = NPURuntime(model_path)
    
    # 3. 准备输入
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 4. 执行推理
    output = runtime.inference({'input': input_data})
    
    return output

# 使用 OpenVINO 部署到 Intel NPU
from openvino.runtime import Core

def deploy_openvino(model_xml, model_bin):
    core = Core()
    
    # 编译模型
    compiled_model = core.compile_model(model_xml, "NPU")
    
    # 创建推理请求
    infer_request = compiled_model.create_infer_request()
    
    # 准备输入
    input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 执行推理
    results = infer_request.infer(inputs=[input_tensor])
    
    return results
```

---

## 📊 性能对比参考

| 量化方案 | 内存占用 | 推理加速 | 精度损失 |
|---------|---------|---------|---------|
| FP32 基准 | 100% | 1.0x | 0% |
| INT8 动态 | ~25% | 2-4x | ±1% |
| INT8 静态 | ~25% | 2-4x | ±2% |
| INT4 静态 | ~12% | 3-5x | ±3-5% |
| INT4 + QAT | ~12% | 3-5x | ±1% |

---

## 🔗 相关工具与框架

- **PyTorch + TorchAO**: https://github.com/pytorch/ao
- **ONNX**: https://onnx.ai/
- **OpenVINO**: Intel NPU 推理 runtime
- **vLLM**: 支持 TorchAO 作为量化后端

---

_文档生成日期: 2026-04-11_