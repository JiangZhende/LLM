import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import torch
from trl import DPOTrainer, DPOConfig
from tiny_dataset import load_dpo_dataset
import matplotlib.pyplot as plt

@dataclass
class ScriptArguments:
    base_model_path: str = field(default="", metadata={"help": "模型路径"})
    tokenizer_path: str = field(default="", metadata={"help": "tokenizer的路径"})
    dataset_dir_or_path: Optional[str] = field(default="", metadata={"help": "训练数据集路径"})
    eval_path: Optional[str] = field(default="", metadata={"help": "评估数据集路径"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "是否只训练1000个样本用于调试"})
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "修复 DDP 中 LM bias/mask buffers 类型问题，详见 https://github.com/huggingface/transformers/issues/22482"
        },
    )
    resume: Optional[bool] = field(default=False, metadata={"help": "是否从 checkpoint 恢复训练"})
    # max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    # max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})


def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, train_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)

    # 1. 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_path,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    # 2. 加载训练数据集
    dpo_dataset = load_dpo_dataset(
        script_args.dataset_dir_or_path,
        max_length=train_args.max_length,
        sanity_check=script_args.sanity_check,
    )

    train_loader = torch.utils.data.DataLoader(
        dpo_dataset.select(range(10)),
        batch_size=train_args.per_device_train_batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=8,
    )
    # 简单打印示例数据，方便调试
    for i, item in enumerate(train_loader):
        print(item)
        break

    # 3. 加载评估数据集（可选）
    if script_args.eval_path:
        evaluation_strategy = "steps"
        eval_dataset = load_dpo_dataset(
            script_args.eval_path,
            max_length=train_args.max_length,
            sanity_check=script_args.sanity_check,
        )
    else:
        evaluation_strategy = "no"
        eval_dataset = None

    # dpo_config = DPOConfig(
    #     **train_args,
    #     max_prompt_length=script_args.max_prompt_length,
    # )

    # 4. 初始化 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=dpo_dataset.select(range(10)),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # max_prompt_length=.max_prompt_length,
        # max_length=dpo_config.max_length,
    )

    # 5. 训练
    dpo_trainer.train(resume_from_checkpoint=script_args.resume)

    # 6. 保存模型
    dpo_trainer.save_model()

    # 7. 绘制训练损失曲线
    def plot_loss(save_directory, log_history):
        plt.switch_backend("agg")
        key = "loss"
        steps, metrics = [], []
        for log_entry in log_history:
            if key in log_entry:
                steps.append(log_entry["step"])
                metrics.append(log_entry[key])

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", label="loss")
        plt.title(f"Training {key} Curve")
        plt.xlabel("Step")
        plt.ylabel(key.capitalize())
        plt.legend()
        figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}.png")
        plt.savefig(figure_path, format="png", dpi=100)
        print(f"Loss curve saved to {figure_path}")

    plot_loss(train_args.output_dir, dpo_trainer.state.log_history)

if __name__ == "__main__":
    main()
