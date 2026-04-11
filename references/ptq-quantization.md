# PTQ (Post-Training Quantization) 调研报告

> 调研 PTQ 数学原理及开源项目实践
> 日期: 2026-04-11

---

## 1. PTQ 数学原理

### 1.1 量化公式：浮点数 ↔ 定点数 转换

量化是将高精度浮点数 (FP32/FP16) 映射到低精度整数 (INT8/INT4/FP4) 的过程。

**统一量化公式：**

```
q = round(x / s) + z
```

**反量化公式 (Dequantization)：**

```
x̂ = (q - z) × s
```

其中：
- x: 原始浮点值
- q: 量化后的整数
- s: 缩放因子 (Scale)
- z: 零点 (Zero-point)
- round(): 四舍五入取整

**量化比特位数 b 决定的整数范围：**

- INT8: 256 个值 (-128 到 127 或 0 到 255)
- INT4: 16 个值 (0 到 15)
- INT2: 4 个值 (0 到 3)
- INT1/Binary: 2 个值 (0 或 1)

### 1.2 对称量化 vs 非对称量化

#### 对称量化 (Symmetric Quantization)

零点 z = 0，量化范围关于零对称。

```
q = round(x / s)
x̂ = q × s

s = max(|x_max|, |x_min|) / (2^(b-1) - 1)
```

- b=8 时，s = max(|x|) / 127
- 范围映射: [-127, 127] → [-max, max]

**优点：**
- 实现简单，只需一个 scale 参数
- 计算效率高 (无零偏移校正)
- 硬件友好

**缺点：**
- 如果数据分布不对称，会浪费量化范围
- 误差较大

#### 非对称量化 (Asymmetric Quantization)

零点 z ≠ 0，允许量化范围偏移以适应数据分布。

```
q = round((x - x_min) / s)
s = (x_max - x_min) / (2^b - 1)
z = -round(x_min / s)
```

或等效形式：

```
q = round(x / s) + z
s = (x_max - x_min) / (2^b - 1)
z = -round(x_min / s)
```

**优点：**
- 更充分利用量化范围
- 更精确的量化误差

**缺点：**
- 需要存储额外的 zero-point
- 计算稍复杂

### 1.3 量化误差分析

**量化误差 (Quantization Error)：**

```
error = x̂ - x = (q - x/s) × s = quantize(x) - x
```

**主要误差来源：**

1. **舍入误差 (Round Error)**
   - 最大为 ±s/2
   
2. **截断误差 (Clipping Error)**
   - 当 |x| > max 或 x 超出范围时发生
   - 严重时主导总误差

3. **分布不均匀误差**
   - 非对称分布时对称量化浪费范围

**误差度量指标：**

- MSE (Mean Squared Error): E[(x̂ - x)²]
- SNR (Signal-to-Noise Ratio): 10×log10(Var(x)/Var(error))
- cosine 相似度

### 1.4 Per-tensor vs Per-channel 量化

#### Per-tensor Quantization

整个张量使用单一 scale 和 zero-point。

```
s = max(|X|) / (2^(b-1) - 1)
z = 0  (对称) 或计算的 z (非对称)
```

**特点：**
- 压缩率高
- 计算简单
- 精度损失较大 (特に权重分布跨度过大)

#### Per-channel Quantization

每个输出通道有独立的 scale (通常用于卷积/线性层)。

```
W: [out_channels, in_channels]
s = [s_1, s_2, ..., s_out_channels]
z = [z_1, z_2, ..., z_out_channels]
```

**特点：**
- 精度损失小
- 压缩率略低 (需要存储更多 scale)
- 推理时需要特殊 kernel 支持

**适用场景：**
- 权重分布差异大
- ReLU 激活通常 per-tensor 即可
- 权重通常 per-channel 更优

---

## 2. 校准算法 (Calibration)

校准是确定量化参数 (s, z) 的过程，需要校准数据集。

### 2.1 MinMax 量化

最简单的方法，使用数据的绝对最大值。

```
x_min = min(X)
x_max = max(X)
s = (x_max - x_min) / (2^b - 1)
z = -round(x_min / s)
```

对称版本：

```
s = max(|x_min|, |x_max|) / (2^(b-1) - 1)
z = 0
```

**特点：**
- 实现简单
- 对异常值敏感，可能浪费范围

### 2.2 Percentile (MSE) 量化

基于 MSE 最优或百分位数的裁剪方法。

**Percentile 方法：**

```
s = (percentile(X, p) - percentile(X, 100-p)) / (2^b - 1)
z = -round(percentile(X, 100-p) / s)
```

典型 p 值: 99.9%, 99.99%

**MSE 最优方法：**

Grid search 寻找最优裁剪范围 [α, β]：

```
min_mse = min E[(quantize(x, α, β) - x)²]
s.t. α ≤ 0 ≤ β
```

### 2.3 EQ (Entropy) 量化

基于 KL 散度最小化，保持量化后分布的信息熵。

```
min D_KL(P(x) || Q(q))
s.t. Q(q) = dequantize(quantize(x, s, z))
```

**实现：**
- 计算原始分布 P 和候选量化分布 Q
- 选择使 KL 散度最小的 scale

### 2.4 混合精度量化

不同层使用不同比特位数，平衡精度与压缩。

**自动混合精度方法：**
1. 逐层测试不同精度
2. 计算每层敏感度 (如梯度或 Hessian)
3. 根据敏感度分配比特位数
4. 迭代优化

**常见策略：**
-  embeddings, LM head → 4-bit 或 8-bit
-  attention → 4-bit
-  FFN → 2-3 bit

---

## 3. 量化感知训练 (QAT) 对比

### 3.1 Forward Fake Quantization

Forward 时模拟量化效果：

```
def fake_quant(x, scale, zero_point, bits):
    q = clamp(round(x / scale) + zero_point, 0, 2^bits - 1)
    return (q - zero_point) * scale
```

**特点：**
- Forward pass 注入量化噪声
- 模型学习适应量化
- 需要完整训练数据

### 3.2 Backward Straight-Through Estimator (STE)

Backward 时绕过伪量化节点的梯度近似：

```
def forward(x):
    q = fake_quant(x)
    return q  # 无梯度

def backward(grad):
    return grad  # STE: 直接传递梯度
```

即：

```
∂L/∂x = ∂L/∂q  (STE 近似)
```

而不是：

```
∂L/∂x = ∂L/∂q × ∂q/∂x (实际为0，因为 round/clamp 不可导)
```

### 3.3 PTQ vs QAT 对比

| 特性 | PTQ | QAT |
|------|-----|-----|
| 训练 | 无需 | 需要完整训练 |
| 数据 | 校准数据 (少) | 训练数据 (多) |
| 时间 | 快 (分钟级) | 慢 (小时级) |
| 精度 | 一般 | 更优 |
| 实现 | 简单 | 复杂 |

**选择建议：**
- PTQ: 快速部署，数据有限
- QAT: 精度优先，有足够资源

---

## 4. 开源项目实践

### 4.1 GPTQ 原理与实现

**GPTQ: Accurate Post-Training Quantization for LLMs**

**核心思想：**
- Layer-wise 量化
- 逐输出通道最优权重近似
- 全局信息 (All-to-All) 减少误差

**算法：**

```python
# 简化伪代码
for layer in layers:
    W, X = get_weights_activations(layer)
    Q = quantize(W, bits)
    
    # 误差补偿
    error = W - dequantize(Q)
    # 将误差反传并吸收
    X = X + error @ pseudo_inverse(X)
```

**关键特性：**
- 逐列 (per-channel) 量化
- 使用 Hessian 近似 (Fisher information)
- 支持 2/3/4-bit 量化
- 典型配置: 4-bit, group_size=128

**实现库：**
- `autogptq`: https://github.com/AutoGPTQ/AutoGPTQ
- `gptqmodel`: https://github.com/ksantoscruz/gptqmodel

### 4.2 AWQ 原理与实现

**AWQ: Activation-aware Weight Quantization**

**核心思想：**
- 基于激活分布而非权重分布确定量化参数
- 保护重要权重 (高激活通道)
- 逐通道 (per-channel) 缩放

**算法：**

```python
# 计算每个通道的重要性 (基于激活)
importance = mean(|W| * mean(|X|, dim=0))

# 缩放权重使小值更小，大值更大
# 量化更容易保护重要权重
for ch in channels:
    s_ch = importance[ch] ** alpha  # alpha 通常 0.5-0.7
    W_scaled[:, ch] = W[:, ch] / s_ch
    
quantize(W_scaled)
```

**关键特性：**
- 激活感知的缩放
- 仅 weight 量化，不量化激活
- 典型配置: 4-bit, group_size=128

**实现库：**
- `llm-awq`: https://github.com/ServiceNow/Aggregation-Via-Linear-Operators/awq

### 4.3 BitAndBytes 使用

**BitAndBytes: 8-bit and 4-bit 量化库**

**特点：**
- Hugging Face Transformers 原生支持
- NF4 (Normal Float 4) 格��
- 双重量化 (Double Quantization)
- 混合 Int8/FP4

**使用示例：**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit 量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config
)

# 4-bit 量化 (NF4)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage="float16"
)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config
)
```

**NF4 格式：**
- 针对正态分布优化
- 动态范围非均匀划分
- 比线性 INT4 更好保持精度

### 4.4 llama.cpp 量化流程

**GGUF 量化格式：**

- Q2_K: 2-bit (type 0)
- Q3_K_S, Q3_K_M, Q3_K_L: 3-bit
- Q4_0, Q4_1, Q4_K_S, Q4_K_M: 4-bit
- Q5_0, Q5_1: 5-bit
- Q6_K: 6-bit
- Q8_0: 8-bit (接近 FP16)

**量化流程：**

```bash
# 下载并量化模型
./quantize /path/to/model.gguf /path/to/model-Q4_K_M.gguf Q4_K_M
```

**k-quants 实现原理：**

```c
// ggml-quants.c 中的实现
// Q4_K 使用 128 个权重为一组
// scale 存储每个 block，bits 存储 quantized weights
typedef struct {
    uint8_t qs[16];    // 4-bit * 32 = 128 bits = 16 bytes
    float dequantized[32];  // k-quants 需要
    float scales[2];
} block_q4_k;
```

**关键特性：**
- 混合量化 (mix of bits per block)
- k-quants 使用 lookup table 加速
- 支持 CPU + GPU 推理

---

## 5. 代码示例

### 5.1 PyTorch PTQ 量化代码

#### 5.1.1 动态量化 (Dynamic PTQ)

```python
import torch
import torch.nn as nn
import torch.quantization

# 动态量化 (权重量化，激活动态)
model = MyModel()
model.eval()
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)

# 校准
with torch.no_grad():
    for data in calibration_loader:
        model(data)

# 转换为量化模型
torch.quantization.convert(model, inplace=True)
model.eval()
```

#### 5.1.2 静态量化 (Static PTQ)

```python
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DequantStub

# 定义量化配置
qconfig = torch.quantization.QConfig(
    activation=torch.quantization.placeholderobserver,
    weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
)

class QLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1, out_features))
        self.zero_point = nn.Parameter(torch.zeros(1, out_features, dtype=torch.int32))
        
    def forward(self, x):
        # 量化权重到 int8
        w_q = torch.quantize_per_tensor(
            self.weight, 
            self.scale, 
            self.zero_point,
            torch.qint8
        )
        # 也量化输入
        x_q = torch.quantize_per_tensor(
            x,
            self.scale,
            self.zero_point,
            torch.qint8
        )
        return torch.ops.quantized.linear(
            x_q, w_q, self.scale, self.zero_point
        )
```

#### 5.1.3 自定义 Observer

```python
import torch
from torch.quantization.observer import Observer

class PercentileObserver(Observer):
    def __init__(self, percentile=99.9):
        super().__init__()
        self.percentile = percentile
        
    def forward(self, x):
        x_flat = x.flatten()
        dim = x_flat.shape[0]
        at = int(dim * self.percentile / 100)
        sorted_x = torch.sort(x_flat)[0]
        val = sorted_x[at] if at < dim else sorted_x[-1]
        
        self.register_buffer('min_val', torch.tensor(x.min()))
        self.register_buffer('max_val', torch.tensor(val))
```

#### 5.1.4 完整 PTQ 实现示例

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

class PTQQuantizer:
    """完整的 PTQ 量化器"""
    
    def __init__(self, model: nn.Module, bits: int = 8):
        self.model = model
        self.bits = bits
        self.calibration_data: List[torch.Tensor] = []
        self.qparams: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.activation_stats: Dict[str, Dict] = {}
    
    def collect_stats(self, data_loader, num_samples: int = 100):
        """收集激活统计信息用于校准"""
        self.model.eval()
        self.activation_stats = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_samples:
                    break
                    
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                # Hook 收集每层激活统计
                self._register_hooks()
                self.model(x)
                
        self._calculate_qparams()
    
    def _register_hooks(self):
        """注册 hooks 收集激活统计"""
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = {
                        'min': [],
                        'max': []
                    }
                
                # 获取输出激活
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                else:
                    act = output[0].detach() if isinstance(output, (tuple, list)) else output
                
                self.activation_stats[name]['min'].append(act.min())
                self.activation_stats[name]['max'].append(act.max())
            
            return hook
        
        # 注册所有可量化层
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        return handles
    
    def _calculate_qparams(self):
        """从统计信息计算量化参数"""
        for name, stats in self.activation_stats.items():
            all_min = torch.stack(stats['min'])
            all_max = torch.stack(stats['max'])
            
            x_min = all_min.min()
            x_max = all_max.max()
            
            # 非对称量化参数
            scale = (x_max - x_min) / (2 ** self.bits - 1)
            zero_point = -round(x_min / scale)
            zero_point = torch.clamp(zero_point, 0, 2 ** self.bits - 1)
            
            self.qparams[name] = (scale, zero_point)
    
    def apply_quantization(self):
        """应用量化到模型"""
        from torch.quantization import QuantStub, DequantStub
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.qparams:
                scale, zero_point = self.qparams[name]
                
                # 替换为量化版本
                qlinear = QuantizedLinear(
                    module.in_features,
                    module.out_features
                )
                qlinear.set_quant_params(scale, zero_point)
                qlinear.weight.data = module.weight.data
                if module.bias is not None:
                    qlinear.bias.data = module.bias.data
                    
                # 替换原模块
                parent = self._get_parent(name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, qlinear)
    
    def _get_parent(self, name: str):
        """获取父模块"""
        parts = name.split('.')
        module = self.model
        for part in parts[:-1]:
            module = getattr(module, part)
        return module


class QuantizedLinear(nn.Module):
    """量化版 Linear 层"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scale = nn.Parameter(torch.ones(1, out_features))
        self.zero_point = nn.Parameter(torch.zeros(1, out_features, dtype=torch.long))
    
    def set_quant_params(self, scale: torch.Tensor, zero_point: torch.Tensor):
        """设置量化参数"""
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化输入
        x_q = torch.quantize_per_tensor(
            x, self.scale, self.zero_point, torch.qint8
        )
        # 量化矩阵乘法
        return torch.ops.quantized.linear(
            x_q, 
            torch.quantize_per_tensor(self.weight, self.scale, self.zero_point, torch.qint8),
            self.scale, self.zero_point
        )
```

#### 5.1.5 使用示例

```python
# 创建模型
model = MyModel().cuda().eval()

# 创建量化器
quantizer = PTQQuantizer(model, bits=8)


# 收集校准数据 (需要真实的推理数据)
calib_loader = DataLoader(calib_dataset, batch_size=32)
quantizer.collect_stats(calib_loader, num_samples=100)

# 应用量化
quantizer.apply_quantization()

# 测试量化后的模型
model.eval()
with torch.no_grad():
    output = model(input_tensor.cuda())
```

### 5.2 ONNX Runtime 量化

#### 5.2.1 动态量化

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态 int8 量化
# 权重在推理时动态量化，激活使用 FP16
quantize_dynamic(
    input_model_path='model.onnx',
    output_model_path='model_dynamic_quant.onnx',
    weight_type=QuantType.QInt8,
    # 优化选项
    op_types_to_quantize=['MatMul', 'Gemm', 'Add', 'Mul', 'Conv'],
    reduce_range=True,  # 使用 reduced range (127 instead of 255)
    optimize_model=True
)
```

#### 5.2.2 静态量化

```python
from onnxruntime.quantization import (
    quantize_static, 
    CalibrationDataReader, 
    QuantType,
    QuantFormat
)
import numpy as np
from torch.utils.data import DataLoader

class ONNXCalibDataReader(CalibrationDataReader):
    """ONNX Runtime 校准数据读取器"""
    
    def __init__(self, data_loader, input_names):
        self.data_loader = data_loader
        self.input_names = input_names
        self.iter = iter(data_loader)
    
    def get_next(self):
        try:
            batch = next(self.iter)
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            # 转换为 numpy
            if hasattr(data, 'numpy'):
                data = data.numpy()
            
            # 构建输入字典
            return {self.input_names[0]: data}
        except StopIteration:
            return None

# 静态 int8 量化
def quantize_onnx_model(
    input_path: str,
    output_path: str,
    calib_data_loader: DataLoader,
    input_name: str = 'input',
    bits: int = 8
):
    """ONNX 模型静态量化"""
    
    calib_reader = ONNXCalibDataReader(calib_data_loader, [input_name])
    
    # 激活类型
    activation_type = QuantType.QInt8 if bits == 8 else QuantType.QUInt8
    
    quantize_static(
        input_model_path=input_path,
        output_model_path=output_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        # QDQ 格式 (Quantize-Dequantize 操作符)
        quant_format=QuantFormat.QDQ,
        # Per-tensor 还是 Per-channel
        per_channel=False,
        # 减少范围
        reduce_range=False,
        # 校准方法
        calibrate_method=1,  # 0=MinMax, 1=Entropy, 2=Percentile
        # 节点白名单
        nodes_to_quantize=['MatMul', 'Gemm', 'Conv', 'Add', 'Mul']
    )
    print(f"量化完成: {output_path}")
```

#### 5.2.3 自定义量化算子

```python
from onnx import numpy_patch, helper, TensorProto
from onnxruntime.quantization import quantize_utils

# 自定义量化 MatMul
def quantized_matmul(
    A, B, scale_a, scale_b, out_scale):
    """量化矩阵乘法"""
    # 解量化
    A_fp = (A.astype(np.int32) - 128) * scale_a
    B_fp = (B.astype(np.int32) - 128) * scale_b
    
    # 矩阵乘法
    C_fp = np.matmul(A_fp, B_fp)
    
    # 重新量化
    C_int = (C_fp / out_scale).round().astype(np.int32) + 128
    return C_int

# 创建自定义 ONNX 量化节点
def make_quant_matmul_node(
    A_name, B_name, output_name,
    scale_a, scale_b, out_scale
):
    """创建量化 MatMul 节点"""
    
    # QuantizeLinear for A
    quant_a = helper.make_node(
        'QuantizeLinear',
        inputs=[A_name, f'{A_name}_scale', f'{A_name}_zero'],
        outputs=[f'{A_name}_quant'],
        name=f'QuantizeLinear_{A_name}'
    )
    
    # QuantizeLinear for B
    quant_b = helper.make_node(
        'QuantizeLinear',
        inputs=[B_name, f'{B_name}_scale', f'{B_name}_zero'],
        outputs=[f'{B_name}_quant'],
        name=f'QuantizeLinear_{B_name}'
    )
    
    # MatMul (使用量化后数据)
    matmul = helper.make_node(
        'MatMul',
        inputs=[f'{A_name}_quant', f'{B_name}_quant'],
        outputs=[f'{output_name}_quant'],
        name=f'MatMul_{output_name}'
    )
    
    # DequantizeLinear
    dequant = helper.make_node(
        'DequantizeLinear',
        inputs=[f'{output_name}_quant', f'{output_name}_scale', f'{output_name}_zero'],
        outputs=[output_name],
        name=f'DequantizeLinear_{output_name}'
    )
    
    return [quant_a, quant_b, matmul, dequant]
```

#### 5.2.4 使用示例

```python
import onnx

# 1. 导出 PyTorch 模型到 ONNX
def export_to_onnx(model, input_tensor, output_path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            output_path,
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    print(f"导出 ONNX 模型: {output_path}")

# 2. 准备校准数据
def create_calib_dataset(num_samples=100):
    """创建校准数据集"""
    calib_data = []
    for _ in range(num_samples):
        # 生成随机输入
        x = np.random.randn(1, 784).astype(np.float32)
        calib_data.append(x)
    return calib_data

# 3. 执行量化
input_onnx = 'model.onnx'
output_onnx = 'model_quant.onnx'

calib_data = create_calib_dataset()
calib_loader = [(x,) for x in calib_data]

quantize_onnx_model(
    input_path=input_onnx,
    output_path=output_onnx,
    calib_data_loader=calib_loader,
    input_name='input',
    bits=8
)

# 4. 验证量化模型
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    output_onnx,
    sess_options,
    providers=['CPUExecutionProvider']
)

# 运行推理
input_data = np.random.randn(1, 784).astype(np.float32)
output = session.run(None, {'input': input_data})
print(f"量化模型推理成功, 输出形状: {output[0].shape}")
```

**QDQ 格式：**

```
Input -> QuantizeLinear -> [ops] -> DequantizeLinear -> Output
```

### 5.3 GPTQ 实践

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 配置
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc=False,  # 不保存 desc
    # 校准参数
    using_triton=False,
    method="gptq"
)

# 加载模型并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "model_name",
    quantized_model_path="./quantized_model",
    quantize_config=quantize_config
)

# 校准数据量化
model.quantize(training_samples)

# 保存
model.save_quantized("./output")
```

### 5.4 llama.cpp Python 接口

```python
from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./model-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=35  # GPU offload layers
)

# 推理
output = llm(
    "Write a short story",
    max_tokens=256,
    temperature=0.8
)
print(output['choices'][0]['text'])
```

### 5.5 Hugging Face BitsAndBytes

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit NF4 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 6. 参考文献

1. GPTQ: Accurate Post-Training Quantization for LLMs - Frantar et al. (2023)
2. AWQ: Activation-aware Weight Quantization for LLMs - Lin et al. (2024)
3. SmoothQuant: Accurate and Efficient Per-channel Quantization - Xiao et al. (2024)
4. GPTQ: Accurate Post-Training Quantization for Generative LLMs - https://arxiv.org/abs/2306.03078
5. llama.cpp quantization - https://github.com/ggml-org/llama.cpp
6. BitsAndBytes - https://github.com/TimDettmers/bitsandbytes

---

## 附录 A：详细数学公式补充

### A.1 量化公式的完整展开

**前向量化 (Quantization):**

```
X_int = round(X_f / S) + Z
```

其中：
- `S` = 缩放因子 (Scale)
- `Z` = 零点 (Zero-point)
- `round()` = 四舍五入到整数

**反向反量化 (Dequantization):**

```
X_f = (X_int - Z) × S
```

**对称量化 (Z=0):**

```
# 量化
X_int = round(X_f / S)
X_f = X_int * S

# Scale 计算
S = max(|X_f|) / (2^(bits-1) - 1)
```

**非对称量化 (Z≠0):**

```
# 量化
X_int = round(X_f / S) + Z

# 反量化
X_f = (X_int - Z) * S

# 参数计算
S = (max(X_f) - min(X_f)) / (2^bits - 1)
Z = -round(min(X_f) / S)
```


### A.2 GPTQ 详细原理补充

#### A.2.1 Hessian 矩阵近似

GPTQ 使用 Fisher Information 作为 Hessian 的近似：

```
# 原始目标函数
minimize: ||W - Q(W)||² × H

# 其中 H 是 Hessian 矩阵
H = ∇²L = X · Xᵀ  (输入激活的外积)

# 对角近似 (简化计算)
H_ii ≈ Σ(x_i²)
# 即每个输入通道的累积能量
```

#### A.2.2 误差补偿数学推导

```
# 量化误差
e = W_original - W_quantized

# 补偿到后续列 (All-to-All)
# 使用伪逆
W_future = W_future + e · X_current⁻¹

# 简化形式:
# 对于列 i 的误差，分配到剩余列 j
# ΔW_j = (eᵀ · x_i) / (x_iᵀ · x_i) × x_j
```

#### A.2.3 逐列量化算法

```python
def gptq_quantize_column(W_col, H_diag, bits):
    """
    单列 GPTQ 量化
    
    W_col: 权重列向量 [out_ch]
    H_diag: Hessian 对角近似 [out_ch]
    bits: 量化位数
    """
    # 1. 计算最优 scale
    max_abs = W_col.abs().max()
    scale = max_abs / (2**(bits-1) - 1)
    
    # 2. 基础量化
    W_int = (W_col / scale).round().clamp(-127, 127)
    
    # 3. 基于 Hessian 的误差修正
    error = W_col - W_int * scale
    
    # 4. 误差加权
    weighted_error = error * H_diag
    
    return W_int * scale, weighted_error
```


### A.3 AWQ 详细原理补充

#### A.3.1 Activation 权重重要性度量

```
# 输入通道重要性
importance_in[i] = mean(|X[:, i]|)

# 输出通道重要性
importance_out[j] = mean(|W[j, :]|)


# 组合重要性
importance[i] = importance_in[i] × importance_out.mean()
```


#### A.3.2 权重缩放公式

```
# 计算缩放因子
s_i = importance[i]^α  # α ∈ [0.5, 0.7]

# 缩放权重
W_scaled[i] = W[i] / s_i

# 量化
Q_scaled[i] = quantize(W_scaled[i])

# 反量化 (恢复原始量程)
Q[i] = Q_scaled[i] × s_i
```

---


*End of Document*