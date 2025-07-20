# LoRA训练和推理指南

本指南介绍如何使用LoRA（Low-Rank Adaptation）进行SFT（Supervised Fine-Tuning）训练，以及如何加载和使用训练好的LoRA模型。

## 1. 环境准备

首先安装必要的依赖：

```bash
pip install -r requirement.txt
```

主要依赖包括：
- `peft>=0.4.0` - LoRA实现
- `transformers>=4.30.0` - 模型和tokenizer
- `torch>=2.0.0` - PyTorch
- `accelerate>=0.20.0` - 分布式训练支持

## 2. LoRA训练

### 2.1 训练脚本

使用 `script/sft_lora.sh` 进行LoRA训练：

```bash
# 给脚本添加执行权限
chmod +x script/sft_lora.sh

# 运行训练
./script/sft_lora.sh
```

### 2.2 主要配置参数

在 `script/sft_lora.sh` 中可以调整以下参数：

```bash
# LoRA配置
LORA_ARGS=" \
    --use_lora True \
    --lora_r 16 \              # LoRA rank，控制适配器大小
    --lora_alpha 32 \          # LoRA alpha参数，通常设置为2*r
    --lora_dropout 0.1 \       # LoRA dropout率
    --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
"

# 训练配置
MBS=4                         # 单卡批次大小
GAS=4                         # 梯度累积步数
LR=5e-4                       # 学习率
TRAIN_EPOCHS=3                # 训练轮数
```

### 2.3 训练输出

训练完成后，模型会保存在：
```
outputs/ckpt/sft_lora_tiny_llm_16m_epoch3/
```

该目录包含：
- `adapter_config.json` - LoRA配置
- `adapter_model.bin` - LoRA权重
- `pytorch_model.bin` - 基础模型权重（如果保存了完整模型）

## 3. 模型加载和使用

### 3.1 方式一：使用推理脚本（推荐）

使用 `inference_lora.py` 进行交互式对话：

```bash
python inference_lora.py \
    --base_model outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model \
    --lora_model outputs/ckpt/sft_lora_tiny_llm_16m_epoch3 \
    --system_prompt "你是由李小贱开发的个人助手。" \
    --merge_weights
```

参数说明：
- `--base_model`: 基础模型路径
- `--lora_model`: LoRA模型路径
- `--merge_weights`: 是否合并LoRA权重到基础模型（推荐用于推理）
- `--system_prompt`: 系统提示词
- `--max_length`: 最大输入长度

### 3.2 方式二：编程方式加载

使用 `load_lora_model.py` 中的函数：

```python
from load_lora_model import load_lora_model, load_lora_model_without_merge

# 加载并合并LoRA权重（用于推理）
model, tokenizer = load_lora_model(
    base_model_path="outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model",
    lora_model_path="outputs/ckpt/sft_lora_tiny_llm_16m_epoch3"
)

# 加载但不合并LoRA权重（用于继续训练）
model, tokenizer = load_lora_model_without_merge(
    base_model_path="outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model",
    lora_model_path="outputs/ckpt/sft_lora_tiny_llm_16m_epoch3"
)
```

### 3.3 方式三：直接使用PEFT

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "outputs/ckpt/sft_lora_tiny_llm_16m_epoch3",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 合并权重（可选）
model = model.merge_and_unload()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model",
    trust_remote_code=True
)
```

## 4. 模型保存和加载的区别

### 4.1 训练时保存
训练过程中，LoRA适配器权重会保存在指定的输出目录中，包含：
- `adapter_config.json` - LoRA配置信息
- `adapter_model.bin` - LoRA权重文件

### 4.2 推理时加载
有两种加载方式：

1. **合并方式**（推荐用于推理）：
   - 将LoRA权重合并到基础模型中
   - 生成一个完整的模型，可以独立使用
   - 使用 `model.merge_and_unload()` 方法

2. **分离方式**（用于继续训练）：
   - 保持LoRA适配器与基础模型分离
   - 可以动态切换不同的LoRA适配器
   - 使用 `PeftModel.from_pretrained()` 方法

## 5. 常见问题

### 5.1 内存不足
- 使用 `torch_dtype=torch.bfloat16` 减少内存使用
- 使用 `device_map="auto"` 自动管理设备分配
- 减小批次大小或使用梯度累积

### 5.2 模型加载失败
- 确保基础模型路径正确
- 确保LoRA模型路径包含 `adapter_config.json` 和 `adapter_model.bin`
- 检查模型架构是否兼容

### 5.3 训练效果不佳
- 调整LoRA rank（r参数）
- 调整学习率
- 增加训练数据或训练轮数
- 调整目标模块（target_modules）

## 6. 性能优化建议

1. **LoRA配置优化**：
   - 对于小模型，可以使用较小的rank（如8-16）
   - 对于大模型，可以使用较大的rank（如32-64）
   - alpha参数通常设置为2*r

2. **训练优化**：
   - 使用bf16混合精度训练
   - 使用DeepSpeed ZeRO优化
   - 适当调整批次大小和梯度累积

3. **推理优化**：
   - 合并LoRA权重以提高推理速度
   - 使用适当的生成参数
   - 考虑使用量化技术进一步压缩模型

## 7. 示例用法

完整的训练和推理流程：

```bash
# 1. 训练LoRA模型
./script/sft_lora.sh

# 2. 测试模型
python inference_lora.py \
    --base_model outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model \
    --lora_model outputs/ckpt/sft_lora_tiny_llm_16m_epoch3

# 3. 开始对话
用户: 你好
助手: 你好！我是由李小贱开发的个人助手，很高兴为您服务。有什么我可以帮助您的吗？
``` 