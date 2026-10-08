#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ENABLE_TIMING="1"
export NCCL_DEBUG="WARN"
CFG="configs/base_comm_attrib.yaml"
LOG_DIR="outputs/comm_attrib/_logs"; mkdir -p "$LOG_DIR"
for strat in ddp fsdp_zero2 fsdp_zero3; do
  for ga in 1 2 4 8; do
    name="qwen3-4b_${strat}_ws8_ga${ga}_bs2_seq2048_rep0"
    echo "START ${name}"
    torchrun --standalone --nproc_per_node=8 -m src.runner \
        --config "$CFG" --models_config configs/models.yaml \
        --model qwen3-4b --strategy "$strat" \
        --world_size 8 --grad_accum_steps "$ga" --rep 0 \
        --experiment_name "$name" \
        --enable_profiler_trace --keep_traces > "${LOG_DIR}/${name}.log" 2>&1
    rc=$?; [[ $rc -eq 0 ]] && echo "DONE  ${name}" || echo "FAIL  ${name} rc=${rc}"
  done
done
echo "ALL_COMM_ATTRIB_COMPLETE"
