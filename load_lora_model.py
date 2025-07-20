import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lora_model(base_model_path, lora_model_path, device="auto"):
    """
    加载LoRA训练的模型
    
    Args:
        base_model_path: 基础模型路径
        lora_model_path: LoRA模型路径（训练保存的路径）
        device: 设备类型，可以是"auto", "cuda", "cpu"
    """
    
    logger.info(f"正在加载基础模型: {base_model_path}")
    logger.info(f"正在加载LoRA适配器: {lora_model_path}")
    
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,  # 使用bfloat16以节省内存
        device_map=device,
        trust_remote_code=True
    )
    
    # 2. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # 3. 合并LoRA权重到基础模型（可选，用于推理）
    logger.info("正在合并LoRA权重...")
    model = model.merge_and_unload()
    
    # 4. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("模型加载完成！")
    
    return model, tokenizer

def load_lora_model_without_merge(base_model_path, lora_model_path, device="auto"):
    """
    加载LoRA模型但不合并权重（用于继续训练或切换不同的LoRA适配器）
    """
    
    logger.info(f"正在加载基础模型: {base_model_path}")
    logger.info(f"正在加载LoRA适配器: {lora_model_path}")
    
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 2. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    # 3. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("模型加载完成（未合并LoRA权重）！")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=512):
    """
    使用加载的模型生成文本
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    # 移动到模型所在设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成配置
    generation_config = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def main():
    # 示例用法
    base_model_path = "outputs/ckpt/ptm_tiny_llm_16m_epoch1/last_ptm_model"  # 基础模型路径
    lora_model_path = "outputs/ckpt/sft_lora_tiny_llm_16m_epoch3"  # LoRA模型路径
    
    # 方式1: 加载并合并LoRA权重（推荐用于推理）
    logger.info("=== 方式1: 加载并合并LoRA权重 ===")
    try:
        model, tokenizer = load_lora_model(base_model_path, lora_model_path)
        
        # 测试生成
        test_prompt = "你是由李小贱开发的个人助手。\n<|user|>\n介绍一下中国。\n<|assistant|>"
        generated_text = generate_text(model, tokenizer, test_prompt)
        print(f"生成的文本: {generated_text}")
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
    
    # 方式2: 加载但不合并LoRA权重（用于继续训练）
    logger.info("\n=== 方式2: 加载但不合并LoRA权重 ===")
    try:
        model, tokenizer = load_lora_model_without_merge(base_model_path, lora_model_path)
        
        # 测试生成
        test_prompt = "你是由李小贱开发的个人助手。\n<|user|>\n介绍一下中国。\n<|assistant|>"
        generated_text = generate_text(model, tokenizer, test_prompt)
        print(f"生成的文本: {generated_text}")
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")

if __name__ == "__main__":
    main() 