#!/usr/bin/env bash
# Confirm-or-refute the noisy small-model cells flagged in RQ2 mechanism panels.
# Isolated output (outputs/ws_confirm/), profiler ON, more PASS-1 steps, 3 reps.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ENABLE_TIMING="1"
export NCCL_DEBUG="WARN"           # quieter than INFO; timing still captured

CFG="configs/base_ws_confirm.yaml"
LOG_DIR="outputs/ws_confirm/_logs"
mkdir -p "$LOG_DIR"

# model strategy world_size
RUNS=(
  "qwen3-0.6b fsdp_zero3 2"
  "qwen3-1.7b fsdp_zero3 2"
  "qwen3-0.6b ddp 2"
  "qwen3-0.6b ddp 8"
  "qwen3-1.7b ddp 8"
)

for spec in "${RUNS[@]}"; do
  read -r model strat ws <<< "$spec"
  for rep in 0 1 2; do
    name="${model}_${strat}_ws${ws}_ga1_bs2_seq2048_rep${rep}"
    log="${LOG_DIR}/${name}.log"
    echo "START ${name}"
    torchrun --standalone --nproc_per_node="${ws}" -m src.runner \
        --config "$CFG" \
        --models_config configs/models.yaml \
        --model "$model" --strategy "$strat" \
        --world_size "$ws" --grad_accum_steps 1 --rep "$rep" \
        --experiment_name "$name" \
        --enable_profiler_trace  > "$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then echo "DONE  ${name}"; else echo "FAIL  ${name} rc=${rc} (see ${log})"; fi
  done
done
echo "ALL_RUNS_COMPLETE"
