#!/bin/bash

#启用调试模式
set -x

#查找并终止与给定参数匹配的进程
function killall {
    echo `ps -ef | grep $1 | grep -v grep | awk '{print $2}'`
    ps -ef | grep $1 | grep -v grep | awk '{print $2}'  | xargs kill
}

# export CUDA_VISIBLE_DEVICES=1 #指定GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1 #启用PyTorch的MPS，用于macOS上的GPU加速

N_NODES=1 #节点数量
N_GPUS=1 #每个节点的GPU数量
MBS=32 #单卡bs批次
GAS=1 #梯度累积
GRAD_CLIP=1 #梯度剪裁
RANK=0 #设置当前节点的排名为0
MASTER_ADDR=`hostname -i` #获取当前主机的IP地址作为主节点地址
MASTER_PORT=9902 

LR=3e-4 
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATION=0.05

TRAIN_EPOCHS=1
LOGGING_STEPS=100
CKPT_SAVE_STEPS=10000 #每10000步保存一次检查点

SEED=12
DS_DTYPE="bf16" #DeepSpeed的数据类型为fp32
RESUME="False" #是否从检查点恢复训练

MODE="ptm"
DATASET_DIR_OR_PATH="datasets/pretrain/" #数据集路径
BASE_MODEL_PATH="tinyllm" #设置基础模型路径

DEEPSPEED="True" #是否使用DeepSpeed

MODEL_SIZE="92m"
MODEL_NAME="${MODE}_tiny_llm_${MODEL_SIZE}"
OUTPUT_DIR="outputs/ckpt/${MODEL_NAME}_epoch${TRAIN_EPOCHS}"

mkdir -p $OUTPUT_DIR
TRAIN_LOG="${OUTPUT_DIR}/train_$(date"+%Y%m%d%H%M").log"

TB_DIR="outputs/tensorboard/${MODEL_NAME}_epoch${TRAIN_EPOCHS}" #TensorBoard日志目录
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
    --adam_beta1 0.95 \
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
if [$DEEPSPEED == "True"];then
    Train_ARGS+="\
    --deepspeed ${DS_CONFIG_JSON} \
    "
fi

if [[ $MODEL_SIZE == "16m" ]];then
    HIDDEN_SIZE=120
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=6
    NUM_KEY_VALUE_HEADS=6
    INTERMEDIATE_SIZE=384
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=512
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "42m" ]];then
    HIDDEN_SIZE=288
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=6
    NUM_KEY_VALUE_HEADS=6
    INTERMEDIATE_SIZE=768
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=512
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "92m" ]];then
    HIDDEN_SIZE=512
    NUM_HIDDEN_LAYERS=8
    NUM_ATTENTION_HEADS=8
    NUM_KEY_VALUE_HEADS=8
    INTERMEDIATE_SIZE=1408
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "210m" ]];then
    HIDDEN_SIZE=768
    NUM_HIDDEN_LAYERS=16
    NUM_ATTENTION_HEADS=12
    NUM_KEY_VALUE_HEADS=12
    INTERMEDIATE_SIZE=2048
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "440m" ]];then
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=24
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=16
    INTERMEDIATE_SIZE=2816
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798
elif [[ $MODEL_SIZE == "0.5b" ]];then
    HIDDEN_SIZE=2048
    NUM_HIDDEN_LAYERS=8
    NUM_ATTENTION_HEADS=8
    NUM_KEY_VALUE_HEADS=8
    INTERMEDIATE_SIZE=2816
    ROPE_THETA=10000.0
    MAX_POSITION_EMBEDDINGS=1024
    VOCAB_SIZE=64798

else
    echo "Unsupported model size: ${MODEL_SIZE}"
    exit 1
fi

GPT_ARGS=" \
    --hidden_size ${HIDDEN_SIZE} \
    --num_hidden_layers ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --rope_theta ${ROPE_THETA} \
    --max_position_embeddings ${MAX_POSITION_EMBEDDINGS} \
    --vocab_size ${VOCAB_SIZE} \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
"
    # --mode ${MODE} \
SCRIPT_ARGS=" \
    --dataset_dir_or_path ${DATASET_DIR_OR_PATH} \
    --resume ${RESUME} \
    --base_model_path ${BASE_MODEL_PATH} \
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

ALL_ARGS=" $GPT_ARGS $SCRIPT_ARGS $TRAIN_ARGS "

LAUNCHER="torchrun $DISTRIBUTED_ARGS train/ptm_train.py "

export CMD="$LAUNCHER $ALL_ARGS"
# CMD="$LAUNCHER $ALL_ARGS"
echo ${CMD}

killall ptm_train.py

# 执行训练
$CMD 2>&1 | tee ${TRAIN_LOG}

killall ptm_train.py

echo "train end : ${OUTPUT_DIR}"