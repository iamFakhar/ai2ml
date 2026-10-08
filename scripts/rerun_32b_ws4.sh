#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ENABLE_TIMING="1"
export NCCL_DEBUG="WARN"
CFG="configs/base_big_confirm.yaml"
LOG_DIR="outputs/ws_confirm/_logs"; mkdir -p "$LOG_DIR"
for rep in 0 1 2; do
  name="qwen3-32b_fsdp_zero3_ws4_ga1_bs2_seq2048_rep${rep}"
  echo "START ${name}"
  torchrun --standalone --nproc_per_node=4 -m src.runner \
      --config "$CFG" --models_config configs/models.yaml \
      --model qwen3-32b --strategy fsdp_zero3 \
      --world_size 4 --grad_accum_steps 1 --rep "$rep" \
      --experiment_name "$name" --enable_profiler_trace > "${LOG_DIR}/${name}.log" 2>&1
  rc=$?; [[ $rc -eq 0 ]] && echo "DONE  ${name}" || echo "FAIL  ${name} rc=${rc}"
done
echo "ALL_32B_COMPLETE"
