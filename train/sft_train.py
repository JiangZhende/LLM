import logging
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    set_seed,
)
import datasets
from tiny_dataset import SFTDataset
from peft import LoraConfig, get_peft_model, TaskType
# from glm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
logger = logging.getLogger(__name__)


# @dataclass
# class ModelArguments:
#     hidden_size: Optional[int] = field(
#         default=512,
#         metadata={"help": "hidden_size"}
#     )
#     num_hidden_layers: Optional[int] = field(
#         default=8,
#         metadata={"help": "num_hidden_layers"}
#     )
#     num_attention_heads: Optional[int] = field(
#         default=8,
#         metadata={"help": "num_attention_heads"}
#     )
#     intermediate_size: Optional[int] = field(
#         default=1408,
#         metadata={"help": "intermediate_size"}
#     )
#     rope_theta: Optional[float] = field(
#         default=10000.0,
#         metadata={"help": "rope_theta"}
#     )
#     max_position_embeddings: Optional[int] = field(
#         default=1024,
#         metadata={"help": "max_position_embeddings"}
#     )
#     vocab_size: Optional[int] = field(
#         default=64798,
#         metadata={"help": "vocab_size"}
#     )

@dataclass
class ScriptArguments:
    dataset_dir_or_path: Optional[str] = field(
        default="data/sft",
        metadata={"help": "save sft file dir"}
    )
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "use PyTorch 2.0 to compile the model to be faster"}
    )
    base_model_path: Optional[str] = field(
        default="",
        metadata={"help": "SFT train, the base model path"}
    )
    tokenizer_path: Optional[str] = field(
        default="",
        metadata={"help": "SFT train, the tokenizer base path"}
    )
    # LoRA配置参数
    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "是否使用LoRA训练"}
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "LoRA目标模块，用逗号分隔"}
    )

def data_collator_fn(examples):
    input_ids = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    data_dict = {
        "input_ids": input_ids,
        "labels": labels
    }
    return data_dict

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info(f"Script Args: \n{script_args}\nTraining Args: \n{training_args}")

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu},"
        +f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    set_seed(training_args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        # config=config,
        trust_remote_code=True
    )
    model.config.use_cache=False
    # model.to(device)

    # 应用LoRA配置
    if script_args.use_lora:
        logger.info("正在应用LoRA配置...")
        target_modules = script_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA配置应用完成")

    max_position_embeddings = model.config.max_position_embeddings
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.tokenizer_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=max_position_embeddings
    )
    # tokenizer = ChatGLMTokenizer.from_pretrained(
    #     script_args.base_model_path,
    #     use_fast=False,
    #     trust_remote_code=True,
    #     model_max_length=max_position_embeddings
    # )

    # config = transformers.AutoConfig.from_pretrained(
    #     script_args.base_model_path,
    #     trust_remote_code=True
    # )
    # config.use_cache=False


    ##################
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params/2**20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}, {trainable_params/2**20:.2f}M params")
    #################

    sft_dataset = SFTDataset(
        script_args.dataset_dir_or_path,
        tokenizer,
        max_position_embeddings
    )

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset
    )
    
    trainer.train(script_args.resume)
    trainer.save_model()
    # last_model_dir = os.path.join(training_args.output_dir, "last_sft_model")
    # os.makedirs(last_model_dir, exist_ok=True)
    # tokenizer.save_pretrained(last_model_dir)
    # trainer.save_model(last_model_dir)

if __name__ == "__main__":
    main()
