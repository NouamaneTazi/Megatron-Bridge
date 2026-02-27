#!/bin/bash
# Qwen3-style 128-expert MoE pretraining on 256 GH200 nodes (CSCS Alps)
# ** Router LR experiment variant **
#
# Config: PP=4 EP=4, Nemotron router (sigmoid + seq_aux_loss + expert bias),
#         1 dense first layer, selective recomputation, English DCLM 100B dataset.
#         Separate learning rate for router parameters.
#
# Requires: feat/router-lr branch on Megatron-LM
#
# Usage: SLURM_JOB_ID=1583727 bash prod/pretrain_qwen3_moe_256n_routerlr.sh [extra hydra overrides...]
# Usage: SLURM_JOB_ID=1583818 bash prod/pretrain_qwen3_moe_256n_routerlr.sh [extra hydra overrides...]
set -euo pipefail
export WANDB_API_KEY=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

# ── Paths ────────────────────────────────────────────────────────────────────
MEGATRON_BRIDGE=/iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge
MEGATRON_LM=/iopsstor/scratch/cscs/ntazi/projects/Megatron-LM
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CONTAINER_EDF="/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps2.toml"
PATCHED_TE_SO=/iopsstor/scratch/cscs/ntazi/projects/TransformerEngine/build/cmake/libtransformer_engine.so
DATASET_CACHE=/iopsstor/scratch/cscs/ntazi/datasets/cache
DATA_CONFIG="${SCRIPT_DIR}/data_configs/english_dclm_100b.json"
TRAINING_SCRIPT=examples/recipes/qwen3_moe/pretrain_qwen3_moe.py

# ── Cluster ──────────────────────────────────────────────────────────────────
NNODES=${NNODES:-256}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

# ── Parallelism ──────────────────────────────────────────────────────────────
TP=1
PP=4
EP=4
DP=$((TOTAL_GPUS / (TP * PP * EP)))
PP_LAYOUT="Et|(tt|)*22L"    # 45 layers across 4 PP stages

# ── Model ────────────────────────────────────────────────────────────────────
NUM_LAYERS=45
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=15360        # Dense layer intermediate size
NUM_EXPERTS=128
MOE_FFN_HIDDEN_SIZE=768
MOE_ROUTER_TOPK=7
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=768
# Layer 0 = dense MLP, layers 1-44 = MoE
MOE_LAYER_FREQ=$(python3 -c "print(','.join(['0']+['1']*($NUM_LAYERS-1)))")

# ── Training ─────────────────────────────────────────────────────────────────
MBS=${MBS:-2}
GBS=$((2 * TOTAL_GPUS * MBS))
TRAIN_ITERS=6001
LR=${LR:-6e-4}
# LR=${LR:-1e-3}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SEQ_LENGTH=4096

# ── Router LR ────────────────────────────────────────────────────────────────
ROUTER_LR=${ROUTER_LR:-6e-5}

# ── Data & Tokenizer ─────────────────────────────────────────────────────────
TOKENIZER="alehc/swissai-tokenizer"

# ── Logging ──────────────────────────────────────────────────────────────────
WANDB_PROJECT="qwen3-moe"
RUN_NAME="${NNODES}n_pp${PP}_ep${EP}_gbs${GBS}_mbs${MBS}_warmup_lr_routerlr"
# RUN_NAME="${NNODES}n_pp${PP}_ep${EP}_gbs${GBS}_mbs${MBS}_warmup_lr2_routerlr"
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
CKPT_DIR="/capstor/scratch/cscs/ntazi/checkpoints/${RUN_NAME}"
LOG_DIR="${SCRIPT_DIR}/slurm_logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${DATASET_CACHE}"

# Save this script for reproducibility
cp -f "${BASH_SOURCE[0]}" "${LOG_DIR}/$(basename "${BASH_SOURCE[0]}")-${TIMESTAMP}.sh" 2>/dev/null || true

# ── Build Training Params ────────────────────────────────────────────────────
TRAINING_PARAMS="\
--model 30b \
model.num_layers=${NUM_LAYERS} \
model.hidden_size=${HIDDEN_SIZE} \
model.ffn_hidden_size=${FFN_HIDDEN_SIZE} \
model.num_moe_experts=${NUM_EXPERTS} \
model.moe_ffn_hidden_size=${MOE_FFN_HIDDEN_SIZE} \
model.moe_router_topk=${MOE_ROUTER_TOPK} \
model.moe_shared_expert_intermediate_size=${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} \
'model.moe_layer_freq=[${MOE_LAYER_FREQ}]' \
model.moe_router_fusion=true \
model.tensor_model_parallel_size=${TP} \
model.pipeline_model_parallel_size=${PP} \
model.expert_model_parallel_size=${EP} \
model.sequence_parallel=false \
model.seq_length=${SEQ_LENGTH} \
model.cuda_graph_impl=none \
model.cuda_graph_scope=null \
model.recompute_granularity=selective \
model.recompute_method=null \
model.recompute_num_layers=null \
'model.recompute_modules=[moe_act,moe,layernorm]' \
model.overlap_moe_expert_parallel_comm=true \
model.deallocate_pipeline_outputs=true \
model.gradient_accumulation_fusion=true \
model.masked_softmax_fusion=true \
model.attention_softmax_in_fp32=false \
model.cross_entropy_fusion_impl=te \
model.cross_entropy_loss_fusion=true \
model.moe_aux_loss_coeff=0.0001 \
model.moe_router_score_function=sigmoid \
model.moe_router_enable_expert_bias=true \
model.moe_router_load_balancing_type=seq_aux_loss \
model.moe_router_dtype=fp32 \
model.moe_shared_expert_overlap=false \
model.moe_router_pre_softmax=true \
model.moe_router_topk_scaling_factor=2.5 \
train.global_batch_size=${GBS} \
train.micro_batch_size=${MBS} \
train.train_iters=${TRAIN_ITERS} \
train.eval_interval=null \
optimizer.lr=${LR} \
optimizer.router_lr=${ROUTER_LR} \
scheduler.lr_warmup_iters=1000 \
scheduler.lr_decay_style=constant \
ddp.align_param_gather=true \
ddp.check_for_nan_in_grad=false \
ddp.disable_symmetric_registration=true \
ddp.nccl_ub=false \
checkpoint.save_interval=${SAVE_INTERVAL} \
checkpoint.save=${CKPT_DIR} \
checkpoint.load=${CKPT_DIR} \
checkpoint.load_optim=true \
logger.log_interval=1 \
logger.wandb_project=${WANDB_PROJECT} \
logger.wandb_exp_name=${RUN_NAME} \
tokenizer.tokenizer_model=${TOKENIZER} \
dataset.path_to_cache=${DATASET_CACHE} \
dataset.dataloader_type=cyclic \
--per-split-data-args-path ${DATA_CONFIG} \
dataset.mock=false"

# Append extra CLI overrides
[[ $# -gt 0 ]] && TRAINING_PARAMS="${TRAINING_PARAMS} $*"

# ── Validate ─────────────────────────────────────────────────────────────────
echo "================================================"
echo "Config: ${RUN_NAME}"
echo "Parallelism: DP=${DP} TP=${TP} PP=${PP} EP=${EP} (${TOTAL_GPUS} GPUs)"
echo "Batch: GBS=${GBS} MBS=${MBS} | Iters: ${TRAIN_ITERS} | LR: ${LR}"
echo "Router LR: ${ROUTER_LR}"
echo "Model: ${NUM_LAYERS}L ${HIDDEN_SIZE}H ${NUM_EXPERTS}E top-${MOE_ROUTER_TOPK}"
echo "Checkpoint: ${CKPT_DIR}"
echo "================================================"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: SLURM_JOB_ID not set. Run with: SLURM_JOB_ID=<jobid> bash $(basename "$0")"
    exit 1
fi

# ── SLURM Setup ──────────────────────────────────────────────────────────────
unset MASTER_ADDR SLURM_JOB_NODELIST

export SLURM_JOB_NODELIST=$(scontrol show job -o "${SLURM_JOB_ID}" \
    | tr ' ' '\n' \
    | awk -F= '$1=="NodeList"{print $2}' \
    | awk '$0 != "(null)" {print; exit}')

MASTER_ADDR="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "Job ${SLURM_JOB_ID} | Nodes: ${SLURM_JOB_NODELIST} | Master: ${MASTER_ADDR}:${MASTER_PORT}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}-${TIMESTAMP}-${SLURM_JOB_ID}.out"
echo "Logging to: ${LOG_FILE}"

set +e

srun \
    --jobid=${SLURM_JOB_ID} \
    --overlap \
    --nodes=${NNODES} \
    --ntasks=${NNODES} \
    --ntasks-per-node=1 \
    --mpi=pmix -ul \
    --environment=${CONTAINER_EDF} \
    --network=disable_rdzv_get \
    bash -lc "
set -x

# ── Distributed Setup ──
export SLURM_GPUS_ON_NODE=${GPUS_PER_NODE}
export SLURM_PROCID=\${SLURM_PROCID}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# ── Code Paths ──
export PYTHONPATH=${MEGATRON_BRIDGE}/src:\${PYTHONPATH:-}
export PYTHONPATH=${MEGATRON_LM}:\${PYTHONPATH:-}
export PP_LAYOUT=\"${PP_LAYOUT}\"

# ── Patched TransformerEngine ──
if [[ -f \"${PATCHED_TE_SO}\" ]]; then
    export LD_PRELOAD=${PATCHED_TE_SO}\${LD_PRELOAD:+:\${LD_PRELOAD}}
fi

# ── NCCL / Communication ──
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NVLINK_DOMAIN_SIZE=4
export UB_SKIPMC=1

# ── TransformerEngine ──
export NVTE_DEBUG=1
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1

# ── PyTorch / CUDA ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Libfabric (must re-export inside container; bash -l overrides inherited values) ──
export FI_MR_CACHE_MONITOR=disabled

# ── HuggingFace ──
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export HF_HOME=/iopsstor/scratch/cscs/ntazi/.cache/huggingface
export HF_HUB_CACHE=/iopsstor/scratch/cscs/ntazi/.cache/huggingface/hub
export HF_TOKEN=${HF_TOKEN:-}

# ── Triton (per-node cache to avoid stale handles on shared FS) ──
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/ntazi/.cache/triton/triton_cache_\${SLURM_PROCID}
mkdir -p \${TRITON_CACHE_DIR}

# ── Wandb ──
export WANDB_API_KEY=\${WANDB_API_KEY:-}
export MEGATRON_SKIP_CONFIG_LOCK=1

cd ${MEGATRON_BRIDGE}

torchrun \
    --nproc_per_node=\${SLURM_GPUS_ON_NODE} \
    --nnodes=${NNODES} \
    --node_rank=\${SLURM_PROCID} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    ${TRAINING_SCRIPT} ${TRAINING_PARAMS}
" 2>&1 | grep --line-buffered -E "^ *(0|$((NNODES - 1))): |mem-reserved-gigabytes" | tee "${LOG_FILE}"
