#!/usr/bin/env bash
set -u

MOSCA_ROOT=/root/autodl-tmp/code/MoSca
DATA_ROOT=/root/autodl-tmp/datasets/dyncheck-mosca
OUTPUT_ROOT="$MOSCA_ROOT/output/sam3_mask_gtcam"
RUN_ROOT="$MOSCA_ROOT/logs/mosca_runs"
RUN_ID=$(date +%Y%m%d_%H%M%S)_dycheck_sam3_mask_gtcam_3gpu
RUN_DIR="$RUN_ROOT/$RUN_ID"
FIT_CFG=./profile/iphone/iphone_fit.yaml
PREP_CFG=./profile/iphone/iphone_prep.yaml
SCENES=(apple block paper-windmill space-out spin teddy wheel)
FORCE_PRECOMPUTE=${FORCE_PRECOMPUTE:-0}
GPUS=(${GPUS:-0 1 2})

mkdir -p "$RUN_DIR" "$OUTPUT_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_dycheck_sam3_mask_gtcam_run.txt"

set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/origs
set -u
cd "$MOSCA_ROOT" || exit 1
export PYTHONPATH=.
export GS_BACKEND=native_add3

echo "run_dir=$RUN_DIR"
echo "scenes=${SCENES[*]}"
echo "gpus=${GPUS[*]}"
echo "fit_cfg=$FIT_CFG"
echo "prep_cfg=$PREP_CFG"
echo "output_root=$OUTPUT_ROOT"
echo "force_precompute=$FORCE_PRECOMPUTE"
echo "started_at=$(date --iso-8601=seconds)"

python scripts/prepare_dycheck_mosca.py "${SCENES[@]}" 2>&1 | tee "$RUN_DIR/prepare.log"

summarize_metrics() {
  local scene="$1"
  (
    flock -x 9
    python scripts/summarize_sam3_mask_metrics.py --output-root "$OUTPUT_ROOT" \
      2>&1 | tee "$RUN_DIR/summary_after_${scene}.log"
  ) 9>"$RUN_DIR/summary.lock"
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local scene_ws="$DATA_ROOT/$scene"
  local scene_log="$RUN_DIR/${scene}.log"
  local scene_status="$RUN_DIR/${scene}.status"
  local precompute_marker="$scene_ws/sam3_dymask/precompute_sam3_mask.done"

  echo "scene=$scene gpu=$gpu start=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"

  if [[ "$FORCE_PRECOMPUTE" == "1" || ! -f "$precompute_marker" || ! -f "$scene_ws/uniform_dep=sensor_bootstapir_tap.npz" || ! -f "$scene_ws/dynamic_dep=sensor_bootstapir_tap.npz" ]]; then
    echo "[$scene] precompute start gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$scene_log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u mosca_precompute.py \
      --ws "$scene_ws" \
      --cfg "$PREP_CFG" \
      --skip_uniform_precompute=True \
      --compute_flow=False \
      2>&1 | tee -a "$scene_log"
    local pre_status=${PIPESTATUS[0]}
    echo "scene=$scene gpu=$gpu precompute_status=$pre_status time=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"
    if [[ "$pre_status" -ne 0 ]]; then
      echo "scene=$scene gpu=$gpu failed_precompute" | tee -a "$RUN_DIR/status.tsv"
      summarize_metrics "$scene"
      return "$pre_status"
    fi
    date --iso-8601=seconds > "$precompute_marker"
  else
    echo "[$scene] precompute skipped; SAM3 mask TAP marker and cached TAP files found" | tee -a "$scene_log"
  fi

  echo "[$scene] reconstruct start gpu=$gpu $(date --iso-8601=seconds)" | tee -a "$scene_log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u mosca_reconstruct.py \
    --ws "$scene_ws" \
    --cfg "$FIT_CFG" \
    --log_root "$OUTPUT_ROOT/$scene" \
    2>&1 | tee -a "$scene_log"
  local recon_status=${PIPESTATUS[0]}
  echo "scene=$scene gpu=$gpu reconstruct_status=$recon_status time=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"

  summarize_metrics "$scene"
  return "$recon_status"
}

run_worker() {
  local gpu="$1"
  shift
  local worker_status=0
  for scene in "$@"; do
    run_scene "$gpu" "$scene" || worker_status=1
  done
  return "$worker_status"
}

summarize_metrics initial

QUEUE_FILE="$RUN_DIR/scene_queue.txt"
printf '%s\n' "${SCENES[@]}" > "$QUEUE_FILE"

pop_scene() {
  local scene=""
  (
    flock -x 8
    if [[ ! -s "$QUEUE_FILE" ]]; then
      exit 1
    fi
    scene=$(head -n 1 "$QUEUE_FILE")
    tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
    mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    printf '%s\n' "$scene"
  ) 8>"$RUN_DIR/queue.lock"
}

run_queue_worker() {
  local gpu="$1"
  local worker_status=0
  local scene=""
  while scene=$(pop_scene); do
    echo "gpu=$gpu dequeued_scene=$scene time=$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/status.tsv"
    run_scene "$gpu" "$scene" || worker_status=1
  done
  echo "gpu=$gpu queue_empty time=$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/status.tsv"
  return "$worker_status"
}

pids=()
for gpu in "${GPUS[@]}"; do
  run_queue_worker "$gpu" 2>&1 | tee "$RUN_DIR/gpu${gpu}_worker.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

echo "finished_at=$(date --iso-8601=seconds) status=$status" | tee -a "$RUN_DIR/status.tsv"
summarize_metrics final
exit "$status"
