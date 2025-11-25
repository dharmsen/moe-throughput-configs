#!/bin/bash

#SBATCH --job-name=fixed_jenia_meglm-qwen3-dev-10B_1.7B
#SBATCH --nodes=256
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --account=laionize
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --output=%j.out
# JUWELS conf

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/p/project1/laionize/harmsen1_jewelsbooster/.cache/huggingface"
TOKENIZER_DIR="${HF_HOME}/hub/models--EleutherAI--gpt-neox-20b/snapshots/c292233c833e336628618a88a648727eb3dff0a7"

MICRO_BATCH_SIZE=1
GAS=8
LR_SCHEDULE="cosine"
LR=3e-4
LR_MIN=3e-5
LR_WARMUP_ITERS=500
TOKENS_TOTAL=350_000_000_000
SEQ_LENGTH=4096
ROTARY_BASE=1000000
SAVE_INTERVAL_ITERS=500
DATA_NAME="comma-0.1"
DATA_SWITCH="GPT-NeoX"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export PROJECT_DIR="/p/data1/mmlaion/shared"
export RUN_DIR="/p/scratch/laionize/harmsen1/moe-scaling/megatron_lm_reference"
mkdir -p $RUN_DIR
export SHARED_CONTAINERS="/p/data1/mmlaion/shared/containers"

MEGATRON_CACHE_BASE="/p/scratch/laionize/harmsen1/container"
MEGATRON_CACHE_FOLDER="${MEGATRON_CACHE_BASE}/${USER}"
mkdir -p ${MEGATRON_CACHE_FOLDER}

export MEGATRON_CACHE="${MEGATRON_CACHE_FOLDER}/MEGATRON_CACHEDIR"
mkdir -p $MEGATRON_CACHE
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p $TENSORBOARD_DIR

export APPTAINER_CACHEDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${MEGATRON_CACHE_FOLDER}/APPTAINER_TMPDIR"


mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

IMAGE=${SHARED_CONTAINERS}/pytorch_24.09-py3.sif

# necessary on JUWELS to handle GLOO CPU communication related errors; on JEDI, this is not required
export GLOO_SOCKET_IFNAME=ib0

# On Leonardo, if experimenting with setting GLOO_SOCKET_IFNAME=ib0, NCCL_SOCKET_IFNAME has to be also set accordingly
export NCCL_DEBUG=INFO

# NCCL settings to improve distributed training stability (handling flipping links, irresponsive nodes, etc)
# waiting for 120s in case nodes become irresponsive giving a chance to recover
export NCCL_IB_TIMEOUT=120

# Training setup
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# ATTENTION: JUWELS still requires manually adding i to MASTER_ADDR
MASTER_ADDR="${MASTER_ADDR}i"
MASTER_IP="$(nslookup "$MASTER_ADDR" |grep -oP '(?<=Address: ).*')"
echo $MASTER_IP
export MASTER_ADDR=$MASTER_IP
MASTER_PORT=12345
NNODES=$SLURM_NNODES

export CUDA_DEVICE_MAX_CONNECTIONS=1


# Data args Starcoder test


DATA="/p/data1/mmlaion/shared/datasets/language"

CHUNKS=0 # number of chunks to go through for DCLM, 0..6
DATA_PATH=""

for ((i = 0; i <= CHUNKS ; i++  ))
do 
    PART_PATH="$DATA/tokenized/${DATA_NAME}/${DATA_SWITCH}/merged_${i}"
    DATA_PATH="${DATA_PATH} ${PART_PATH}"
done

TOKENIZER_MODEL="${TOKENIZER_DIR}"
TOKENIZER_TYPE="HuggingFaceTokenizer"
VOCAB_FILE="${TOKENIZER_MODEL}/vocab.json"
MERGE_FILE="${TOKENIZER_MODEL}/merges.txt"

# Data args C4 reference
# DATA=/p/data1/mmlaion/text-data/RedPajama-Data-1T/c4

DATA_NUM_WORKERS=4
# DATA_ARGS=(
#     --data-path $DATA_PATH 
#     --tokenizer-model $TOKENIZER_MODEL
#     --vocab-file $VOCAB_FILE
#     --merge-file $MERGE_FILE

#     --split 989,10,1
#     --num-workers $DATA_NUM_WORKERS
# )
DATA_ARGS=(
    --tokenizer-model "$TOKENIZER_MODEL"
    # --tokenizer-type "$TOKENIZER_TYPE"
    --vocab-file "$TOKENIZER_MODEL/vocab.json"
    --merge-file "$TOKENIZER_MODEL/merges.txt"
    # --mock-data
    --data-path $DATA_PATH
    --split 989,10,1
    --make-vocab-size-divisible-by 128
    --num-workers $DATA_NUM_WORKERS
)


# GPT args 1b
NUM_LAYERS=18
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
# FFN_HIDDEN_SIZE=6144


# MAX_POSITION_EMBEDDINGS=${SEQ_LENGTH}
MAX_POSITION_EMBEDDINGS=32768

GPT_MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    # moe
    --moe-ffn-hidden-size 1408
    --num-experts 64
    --moe-router-topk 8
    --moe-router-dtype fp64
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-force-load-balancing
    # --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTN_HEADS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
)


TP=1
# PP=1
PP=2
EP=4

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP 
	--pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --sequence-parallel
    --context-parallel-size 1
)
NUM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_JOB_NUM_NODES))

echo "SLURM_GPUS_PER_NODE: " $SLURM_GPUS_PER_NODE
echo "SLURM_JOB_NUM_NODES: " $SLURM_JOB_NUM_NODES 
echo "NUM_GPUS: " $NUM_GPUS
GLOBAL_BATCH_SIZE=$(((NUM_GPUS*MICRO_BATCH_SIZE*GAS)/TP))
# GLOBAL_BATCH_SIZE=512
TOKENS_GLOBAL_BATCH_SIZE=$((SEQ_LENGTH*GLOBAL_BATCH_SIZE))

echo "MICRO_BATCH_SIZE: " $MICRO_BATCH_SIZE
echo "GRADIENT_ACCUMULATION_STEPS: " $GAS
echo "GLOBAL_BATCH_SIZE: " $GLOBAL_BATCH_SIZE
echo "SEQUENCE LENGTH: " ${SEQ_LENGTH}
echo "TOKENS_GLOBAL_BATCH_SIZE: " ${TOKENS_GLOBAL_BATCH_SIZE}


CHECKPOINT_FORMAT="torch_dist"

if (( TP > 1 || PP > 1 )); then 

    CHECKPOINT_FORMAT="torch_dist"

fi

TOTAL_TOKENS_NUM=${TOKENS_TOTAL} # 300B, 50B tokens
BILLION_NUM=1000000000 # 1B
TOKENS_BILLION=$(( TOTAL_TOKENS_NUM / BILLION_NUM ))
TOTAL_TOKENS_LABEL="${TOKENS_BILLION}B"

COOLDOWN_FRACTION=1/5
# ceil for total train iterations, TRAIN_ITERS = TOTAL_TOKENS_NUM / SEQ_LENGTH / GLOBAL_BATCH_SIZE
# TRAIN_ITERS=$(((${TOTAL_TOKENS_NUM} + (${SEQ_LENGTH} * ${GLOBAL_BATCH_SIZE}) - 1)/(${SEQ_LENGTH}*${GLOBAL_BATCH_SIZE})))
TRAIN_ITERS=2000
LR_DECAY_ITERS=$TRAIN_ITERS
LR_WSD_DECAY_ITERS=$((${TRAIN_ITERS} * ${COOLDOWN_FRACTION}))

SAVE_INTERVAL=${SAVE_INTERVAL_ITERS}
EVAL_INTERVAL=5000
LOG_INTERVAL=1
EVAL_ITERS=100

echo "TOTAL TOKENS: " $TOTAL_TOKENS_NUM
echo "TOTAL TOKENS LABEL: " $TOTAL_TOKENS_LABEL
echo "TRAIN_ITERS: " $TRAIN_ITERS
echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR_DECAY_ITERS: " $LR_DECAY_ITERS
echo "LR_WSD_DECAY_ITERS: " $LR_WSD_DECAY_ITERS

LR_DECAY_STYLE=${LR_SCHEDULE}
LR_WSD_DECAY_STYLE="linear"

echo "LR_WARMUP_ITERS: " $LR_WARMUP_ITERS
echo "LR: " $LR

ROTARY_PERCENT=1.0

NORM_EPSILON=1e-6
INIT_METHOD_STD=0.02

# Training args
TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 1e-1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95 
    --adam-eps 1e-8
    --init-method-std ${INIT_METHOD_STD}
    --clip-grad 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --lr-decay-style ${LR_DECAY_STYLE}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr ${LR}
    --min-lr ${LR_MIN}
    # --lr-wsd-decay-style ${LR_WSD_DECAY_STYLE}
    # --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS}
    --data-cache-path $MEGATRON_CACHE
    --use-flash-attn
    --attention-softmax-in-fp32
    --bf16
    --qk-layernorm  
    --tensorboard-dir $TENSORBOARD_DIR
    --tensorboard-queue-size 5
    --ckpt-format $CHECKPOINT_FORMAT
    --position-embedding-type rope
    --rotary-base ${ROTARY_BASE}
    # --rotary-percent ${ROTARY_PERCENT}
    --normalization RMSNorm
    --norm-epsilon ${NORM_EPSILON}
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --distributed-backend nccl 
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    # --recompute-activations
    --kv-channels 128
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 30
)

CHECKPOINT_PATH="${RUN_DIR}/checkpoints"
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
EXP_LABEL="open-sci-ref_model-1b_data-${DATA_NAME}_samples-${TOTAL_TOKENS_LABEL}_global_bs-${GLOBAL_BATCH_SIZE}_context-${SEQ_LENGTH}_rotary-${ROTARY_BASE}_schedule-${LR_DECAY_STYLE}_lr-${LR}_warmup-${LR_WARMUP_ITERS}_machine-JUWELS-TEST"

CHECKPOINT_PATH="$CHECKPOINT_PATH/${EXP_LABEL}"

mkdir -p $CHECKPOINT_PATH
TENSORBOARD_LOGS_PATH="$CHECKPOINT_PATH/tensorboard"
mkdir -p $TENSORBOARD_LOGS_PATH


# Eval and logging args
EVAL_AND_LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --save-interval ${SAVE_INTERVAL} 
    --eval-interval ${EVAL_INTERVAL} 
    --log-throughput
    --log-progress
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters ${EVAL_ITERS}
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Command
CMD="pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    "

# Distributed args
DISTRIBUTED_ARGS=(
    --nproc-per-node $GPUS_PER_NODE 
    --nnodes $NNODES
)


LAUNCHER="singularity exec \
    --nv \
    $IMAGE \
   python -u -m torch.distributed.run \
    ${DISTRIBUTED_ARGS[@]} \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend static \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD


SRUN_ARGS=" \
    --wait=60 --cpus-per-task=48 --threads-per-core=1 \
    --kill-on-bad-exit=1"

# MEGATRON_PATH="/p/project1/laionize/marianna/megatron/Megatron-LM"

# MEGATRON_PATH="/p/data1/mmlaion/shared/repos/Megatron-LM"
MEGATRON_PATH="/p/project1/laionize/harmsen1_jewelsbooster/moe-scaling/Megatron-LM-Snellius/Megatron-LM"


srun $SRUN_ARGS \
    --jobid $SLURM_JOB_ID \
    bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

