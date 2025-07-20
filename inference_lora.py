#!/usr/bin/env python3
"""
LoRA模型推理脚本
使用方法: python inference_lora.py --base_model <基础模型路径> --lora_model <LoRA模型路径>
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(base_model_path, lora_model_path, merge_weights=True):
    """加载LoRA模型"""
    logger.info(f"加载基础模型: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"加载LoRA适配器: {lora_model_path}")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if merge_weights:
        logger.info("合并LoRA权重到基础模型...")
        model = model.merge_and_unload()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def chat_with_model(model, tokenizer, system_prompt="你是由李小贱开发的个人助手。", max_length=512):
    """与模型对话"""
    print(f"系统提示: {system_prompt}")
    print("开始对话 (输入 'quit' 退出):")
    
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == 'quit':
            break
        
        # 构建完整的对话格式
        full_prompt = f"{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>"
        
        # 编码输入
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码回复
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手的回复部分
        if "<|assistant|>" in response:
            assistant_response = response.split("<|assistant|>")[-1].strip()
        else:
            assistant_response = response.split(user_input)[-1].strip()
        
        print(f"助手: {assistant_response}")

def main():
    parser = argparse.ArgumentParser(description="LoRA模型推理")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--merge_weights", action="store_true", default=True, help="是否合并LoRA权重")
    parser.add_argument("--system_prompt", type=str, default="你是由李小贱开发的个人助手。", help="系统提示词")
    parser.add_argument("--max_length", type=int, default=512, help="最大输入长度")
    
    args = parser.parse_args()
    
    try:
        # 加载模型
        model, tokenizer = load_model(args.base_model, args.lora_model, args.merge_weights)
        
        # 开始对话
        chat_with_model(model, tokenizer, args.system_prompt, args.max_length)
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

if __name__ == "__main__":
    main() 