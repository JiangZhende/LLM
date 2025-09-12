# Tiny QWEN

## 本文旨在构建一个小参数量的QWEN模型，用于快速学习模型训练的相关知识。

## 项目结构

.
├── glm3_tokenizer # tokenizer 目录  
├── prepare.sh # 数据集下载  
├── process # 数据处理  
├── requirement.txt  
├── script # 启动脚本. 
├── train # 训练脚本. 

## 使用框架

本项目主要使用HuggingFace的框架transformers和trl进行模型训练。

## 安装依赖

- python3.10
- CUDA 12.4

```bash
pip install -r requirement.txt
```

## 数据准备
预训练 TigerResearch/pretrain_zh

微调 TigerResearch/sft_zh


