import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from transformers import TrainingArguments, HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from trl import DPOTrainer
from tiny_dataset import load_dpo_dataset
@dataclass
class ScriptArguments:
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "the beta parameter for DPO loss"}
    )
    model: str = field(
        default="",
        metadata={"help": "模型路径"}
    )
    tokenizer: str = field(
        default="",
        metadata={"help": "tokenizer的路径"}
    )
    dataset_path: Optional[str] = field(
        default="",
        metadata={"help": "dataset path"}
    )
    eval_path :Optional[str] = field(
        default="",
        metadata={"help": "eval dataset path"}
    )
    resume: Optional[bool] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"}
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "the maximum sequence length"}
    )
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "epoch of training steps"})
    logging_strategy: Optional[str] = field(default="steps",  metadata={"help": "logging_strategy"})
    logging_dir: Optional[str] = field(default="",  metadata={"help": "logging_dir"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def main():
    
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    data_path = "/mnt/cephfs-xiongzhuang/wangdongnian/tiny-llm-zh/data/rm_train/rm_data.jsonl"
    dpo_dataset = load_dpo_dataset(script_args.dataset_dir_or_path, max_length=script_args.max_length, sanity_check=script_args.sanity_check)

    train_loader = torch.utils.data.DataLoader(
        dpo_dataset,
        batch_size=2,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=8,
    )
    for i, item in enumerate(train_loader):
        print(item)
        break

    # 3. Load evaluation dataset
    if script_args.eval_dataset_dir_or_path == "":
        evaluation_strategy = "no"
    else:
        evaluation_strategy = "steps"
        eval_dataset = load_dpo_dataset(script_args.eval_dataset_dir_or_path, max_length=script_args.max_length, sanity_check=script_args.sanity_check)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_dir=script_args.logging_dir,
        logging_strategy=script_args.logging_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="no",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        remove_unused_columns=False,
        run_name=script_args.model_name,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        # project_kwargs={"logging_dir": script_args.output_dir},
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=dpo_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        # data_collator=collator_fn,
    )
    # 6. train
    dpo_trainer.train(script_args.resume)
    

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "last_dpo_model")
    tokenizer.save_pretrained(output_dir)
    dpo_trainer.save_model(output_dir)
    # dpo_trainer.model.save_pretrained(output_dir)