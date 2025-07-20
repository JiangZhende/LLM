import logging
import numpy as np
import os
import glob
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
import torch

import datasets
import transformers
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    
)
from tiny_dataset import PTMDataset
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    hidden_size: Optional[int] = field(
        default=512,
        metadata={"help": "hidden_size"}
    )
    intermediate_size: Optional[int] = field(
        default=2752,
        metadata={"help": "Dimension of the MLP"}
    )
    num_hidden_layers: Optional[int] = field(
        default=8,
        metadata={"help": "num_hidden_layers"}
    )
    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={"help": "transformer num_attention_heads"}
    )
    num_key_value_heads: Optional[int] = field(
        default=8,
        metadata={"help": "num_key_value_heads"}
    )

    hidden_act: Optional[str] = field(
        default="silu",
        metadata={"help": "activation function"}
    )
    rope_theta: Optional[float] = field(
        default=10000.0,
        metadata={"help": "rope_theta"}
    )
    max_position_embeddings: Optional[int] = field(
        default=1024,
        metadata={"help": "max_position_embeddings"}
    )
    vocab_size: Optional[int] = field(
        default=64798,
        metadata={"help": "vocab_size"}
    )

@dataclass
class ScriptArguments:
    dataset_dir_or_path: Optional[str] = field(
        default="dataset/pre_train",
        metadata={"help": "save pretrain *binfile dir"}
    )
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "use PyTorch 2.0 to compile the mode to be faster"}
    )
    base_model_path: Optional[str] = field(
        default="",
        metadata={"help": "SFT train, the base model path"}
    )

def get_bin_files_abs_paths(directory):
    bin_files_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bin"):
                bin_files_paths.append(os.path.abspath(os.path.join(root, file)))
    
    return bin_files_paths


def main():
    parser = HfArgumentParser((ModelArguments, ScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info(f"Model Args: \n{model_args}\nScript Args: \n{script_args}\nTraining Args: \n{training_args}")
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )

    set_seed(training_args.seed)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Qwen2Config(
        vocab_size=model_args.vocab_size,
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        num_attention_heads=model_args.num_attention_heads,
        num_hidden_layers=model_args.num_hidden_layers,
        rope_theta=model_args.rope_theta,
        max_position_embeddings=model_args.max_position_embeddings,
        num_key_value_heads=model_args.num_key_value_heads,
        hidden_act=model_args.hidden_act)
    logger.info(f"Model Config:\n{config}")

    model = Qwen2ForCausalLM(config)
    # model.to(device)

    ###################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params/2**20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}, {trainable_params/2**20:.2f}M params")
    ###################

    data_path_list = get_bin_files_abs_paths(script_args.dataset_dir_or_path)
    logger.info(f"数据路径列表长度: {len(data_path_list)}, 内容: {data_path_list}")
    if len(data_path_list) == 0:
        logger.error("***************NO INPUT DATA**********************")

    train_ds = PTMDataset(data_path_list, max_length=model_args.max_position_embeddings)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds
    )
    # 开始训练
    trainer.train(script_args.resume)
    # trainer.train()
    trainer.save_model()  # 保存模型
    # tokenizer.save_pretrained(output_path)  # 保存分词器
    
    # torch.save(model.state_dict(), "{}/last_model.pth".format(training_args.output_dir))
    # last_model_dir = os.path.join(training_args.output_dir, "last_ptm_model")
    # os.makedirs(last_model_dir, exist_ok=True)
    # model.save_pretrained(last_model_dir, safe_serialization=False)

if __name__ == "__main__":
    main()