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

**动态量化 (Dynamic PTQ)：**

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

**静态量化 (Static PTQ)：**

```python
import torch
import torch.nn as nn

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
        
    def forward(self, x):
        # 量化权重到 int8
        w_q = torch.quantize_per_tensor(
            self.weight, 
            self.scale, 
            0, 
            torch.qint8
        )
        return torch.ops.quantized.linear(
            x, w_q, self.scale, 0
        )
```

**自定义 observer：**

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

### 5.2 ONNX Runtime 量化

**动态量化：**

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态 int8 量化
quantize_dynamic(
    input_model_path='model.onnx',
    output_model_path='model_dynamic_quant.onnx',
    weight_type=QuantType.QInt8,
    # 优化选项
    op_types_to_quantize=['MatMul', 'Gemm', 'Add', 'Mul'],
    reduce_range=True
)
```

**静态量化：**

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# 准备校准数据
def calib_data():
    for data in calib_loader:
        yield {'input': data.numpy()}

# 静态 int8 量化
quantize_static(
    input_model_path='model.onnx',
    output_model_path='model_static_quant.onnx',
    calibration_data_reader=calib_data(),
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    # 量化策略
    quant_format=QuantFormat.QDQ,  # Quantize + Dequantize ops
    per_channel=False,
    reduce_range=False
)
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

*End of Document*