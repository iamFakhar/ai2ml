#!/usr/bin/env bash
# ==============================================================================
# nccl_tests.sh — Hardware BW characterization (design doc §7 Type 3)
# ==============================================================================
# Runs nccl-tests across {2,4,8} GPUs for all-reduce / all-gather /
# reduce-scatter, sweeping message sizes 1M..1G, and emits:
#   - raw per-collective logs
#   - bw_curves.json (consumed by instrumentation/profiler.score_bandwidth)
#
# The JSON is the size-resolved bandwidth curve that turns "FSDP is slower"
# into "FSDP's message sizes achieve only X% of achievable BW" (RQ2 mechanism).
#
# Prereqs:
#   - nccl-tests built at $NCCL_TESTS_DIR (default ~/nccl-tests)
#     git clone https://github.com/NVIDIA/nccl-tests
#     cd nccl-tests && make MPI=0 CUDA_HOME=/usr/local/cuda
#
# Usage:
#   bash analysis/nccl_tests.sh
#   NCCL_TESTS_DIR=/path/to/nccl-tests OUT_DIR=./outputs/nccl_tests bash analysis/nccl_tests.sh
# ==============================================================================

set -euo pipefail

NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-$HOME/nccl-tests}"
OUT_DIR="${OUT_DIR:-./outputs/nccl_tests}"
GPU_COUNTS="${GPU_COUNTS:-2 4 8}"
MIN_BYTES="${MIN_BYTES:-1M}"
MAX_BYTES="${MAX_BYTES:-1G}"
STEP_FACTOR="${STEP_FACTOR:-2}"   # -f 2 : geometric size sweep

mkdir -p "$OUT_DIR/raw"

# Match runtime env so BW numbers reflect the SAME conditions as training.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"

declare -A PERF_BINS=(
  [allreduce]="all_reduce_perf"
  [allgather]="all_gather_perf"
  [reducescatter]="reduce_scatter_perf"
)

echo "======================================================================"
echo "NCCL-TESTS HARDWARE CHARACTERIZATION"
echo "  nccl-tests dir : $NCCL_TESTS_DIR"
echo "  output dir     : $OUT_DIR"
echo "  GPU counts     : $GPU_COUNTS"
echo "  size sweep     : $MIN_BYTES .. $MAX_BYTES (x$STEP_FACTOR)"
echo "======================================================================"

for g in $GPU_COUNTS; do
  for coll in "${!PERF_BINS[@]}"; do
    bin="${PERF_BINS[$coll]}"
    bin_path="$NCCL_TESTS_DIR/build/$bin"
    raw_out="$OUT_DIR/raw/${coll}_g${g}.log"

    if [[ ! -x "$bin_path" ]]; then
      echo "ERROR: missing $bin_path — did you build nccl-tests?" >&2
      exit 1
    fi

    echo ""
    echo ">>> $coll @ ${g} GPUs -> $raw_out"
    "$bin_path" \
        -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" \
        -g "$g" \
        2>&1 | tee "$raw_out"
  done
done

echo ""
echo ">>> Parsing raw logs into $OUT_DIR/bw_curves.json"
python3 "$(dirname "$0")/parse_nccl_tests.py" \
    --raw_dir "$OUT_DIR/raw" \
    --out_json "$OUT_DIR/bw_curves.json"

echo ""
echo "DONE. bw_curves.json ready for bandwidth_efficiency scoring."
echo "Point base.yaml hardware.nccl_bw_curve_path at:"
echo "  $OUT_DIR/bw_curves.json"