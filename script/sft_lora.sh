#!/bin/bash

#启用调试模式
set -x

#查找并终止与给定参数匹配的进程
function killall {
    echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
    ps -ef | grep $1 | grep -v grep | awk '{print $2}'  | xargs kill
}

export PYTORCH_ENABLE_MPS_FALLBACK=1

N_NODES=1
N_GPUS=1
MBS=16 # 单卡批次大小，LoRA训练可以使用更大的批次
GAS=1 # 梯度累积
GRAD_CLIP=1
RANK=0
MASTER_ADDR=`hostname -i`
MASTER_PORT=9903

LR=5e-4 # LoRA训练通常使用更高的学习率
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATION=0.1

TRAIN_EPOCHS=3
LOGGING_STEPS=10
CKPT_SAVE_STEPS=500

SEED=42
DS_DTYPE="bf16" # 使用bf16进行LoRA训练
RESUME="False"

MODE="sft" # SFT模式
DATASET_DIR_OR_PATH="datasets/sft/sft_train/sft_data_test.jsonl" # SFT数据集路径
BASE_MODEL_PATH="outputs/ckpt/ptm_tiny_llm_92m_epoch1/checkpoint-549071" # 预训练模型路径
TOKENIZER_PATH="glm3_tokenizer"

DEEPSPEED="True" # 使用DeepSpeed进行LoRA训练

MODEL_SIZE="92m"
MODEL_NAME="${MODE}_tiny_llm_${MODEL_SIZE}"
OUTPUT_DIR="outputs/ckpt/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"

mkdir -p $OUTPUT_DIR
TRAIN_LOG="${OUTPUT_DIR}/train_$(date +%Y%m%d%H%M).log"

TB_DIR="outputs/tensorboard/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"
mkdir -p $TB_DIR

TRAIN_ARGS=""

DS_CONFIG_JSON=${OUTPUT_DIR}/${MODEL_SIZE}_ds_config.json
ZERO_STAGE=2

if [ $DS_DTYPE = "fp16" ];then
    TRAIN_ARGS+="\
        --fp16 \
        "
    DS_FP16=true
    DS_BF16=false
    GAS_DTYPE=$DS_DTYPE
elif [ $DS_DTYPE = "bf16" ];then
    TRAIN_ARGS+=" \
        --bf16 \
        "
    DS_FP16=false
    DS_BF16=true
    GAS_DTYPE="fp32"
elif [ $DS_DTYPE = "fp32" ];then
    DS_FP16=false
    DS_BF16=false
    GAS_DTYPE="fp32"
fi

cat <<EOT > $DS_CONFIG_JSON
{
  "train_micro_batch_size_per_gpu": $MBS,
  "train_batch_size": "auto",
  "gradient_clipping": ${GRAD_CLIP},
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": ${DS_BF16}
  },
  "data_types": {
    "grad_accum_dtype": "${GAS_DTYPE}"
  },
  "fp16": {
    "enabled": ${DS_FP16},
    "loss_scale": 0,
    "loss_scale_window": 200,
    "hysteresis": 5,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": true,
  "comms_logger": {
      "enabled": true,
      "verbose": false,
      "prof_all": false,
      "debug": false
    },
    "flops_profiler": {
        "enabled": false,
        "profile_step": 30,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
    }
}
EOT

TRAIN_ARGS+=" \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${GAS} \
    --do_train \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --logging_dir ${TB_DIR} \
    --logging_strategy steps \
    --logging_steps ${LOGGING_STEPS} \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm ${GRAD_CLIP} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --learning_rate ${LR} \
    --warmup_ratio ${WARMUP_RATION} \
    --weight_decay 0.01 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps ${CKPT_SAVE_STEPS} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --save_safetensors True \
    --ddp_find_unused_parameters False \
"

if [ $DEEPSPEED == "True" ];then
    TRAIN_ARGS+="\
    --deepspeed ${DS_CONFIG_JSON} \
    "
fi

# LoRA配置参数
LORA_ARGS=" \
    --use_lora False \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
"

SCRIPT_ARGS=" \
    --dataset_dir_or_path ${DATASET_DIR_OR_PATH} \
    --resume ${RESUME} \
    --base_model_path ${BASE_MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH}
"

DISTRIBUTED_ARGS=" \
    --nnodes $N_NODES \
    --nproc_per_node $N_GPUS \
"

if [ "$N_NODES" -ge 2 ]; then
    DISTRIBUTED_ARGS+=" \
        --node_rank $RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
    "
fi

ALL_ARGS=" $LORA_ARGS $SCRIPT_ARGS $TRAIN_ARGS "

LAUNCHER="torchrun $DISTRIBUTED_ARGS train/sft_train.py "

export CMD="$LAUNCHER $ALL_ARGS"
echo ${CMD}

killall sft_train.py

# 执行训练
$CMD 2>&1 | tee ${TRAIN_LOG} 