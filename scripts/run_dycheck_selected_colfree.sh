#!/usr/bin/env bash
set -u

MOSCA_ROOT=/root/autodl-tmp/code/MoSca
DATA_ROOT=/root/autodl-tmp/datasets/dyncheck-mosca
OUTPUT_ROOT="$MOSCA_ROOT/output/colfree"
RUN_ROOT="$MOSCA_ROOT/logs/mosca_runs"
RUN_ID=$(date +%Y%m%d_%H%M%S)_dycheck_selected_colfree
RUN_DIR="$RUN_ROOT/$RUN_ID"
SCENES=(apple block paper-windmill space-out spin teddy wheel)

mkdir -p "$RUN_DIR" "$OUTPUT_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_dycheck_colfree_run.txt"

set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/origs
set -u
cd "$MOSCA_ROOT" || exit 1
export PYTHONPATH=.
export GS_BACKEND=native_add3

echo "run_dir=$RUN_DIR"
echo "scenes=${SCENES[*]}"
echo "cfg=./profile/iphone/iphone_fit_colfree.yaml"
echo "output_root=$OUTPUT_ROOT"
echo "started_at=$(date --iso-8601=seconds)"

python scripts/prepare_dycheck_mosca.py "${SCENES[@]}" 2>&1 | tee "$RUN_DIR/prepare.log"

for scene in "${SCENES[@]}"; do
  scene_ws="$DATA_ROOT/$scene"
  scene_log="$RUN_DIR/${scene}.log"
  scene_status="$RUN_DIR/${scene}.status"
  echo "scene=$scene start=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"

  if [[ ! -f "$scene_ws/uniform_dep=sensor_bootstapir_tap.npz" || ! -f "$scene_ws/dynamic_dep=sensor_bootstapir_tap.npz" ]]; then
    echo "[$scene] precompute start $(date --iso-8601=seconds)" | tee -a "$scene_log"
    CUDA_VISIBLE_DEVICES=0 python -u mosca_precompute.py \
      --ws "$scene_ws" \
      --cfg ./profile/iphone/iphone_prep.yaml \
      2>&1 | tee -a "$scene_log"
    pre_status=${PIPESTATUS[0]}
    echo "scene=$scene precompute_status=$pre_status time=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"
    if [[ "$pre_status" -ne 0 ]]; then
      echo "scene=$scene failed_precompute" | tee -a "$RUN_DIR/status.tsv"
      continue
    fi
  else
    echo "[$scene] precompute skipped; cached TAP files found" | tee -a "$scene_log"
  fi

  echo "[$scene] reconstruct start $(date --iso-8601=seconds)" | tee -a "$scene_log"
  CUDA_VISIBLE_DEVICES=0 python -u mosca_reconstruct.py \
    --ws "$scene_ws" \
    --cfg ./profile/iphone/iphone_fit_colfree.yaml \
    --log_root "$OUTPUT_ROOT/$scene" \
    2>&1 | tee -a "$scene_log"
  recon_status=${PIPESTATUS[0]}
  echo "scene=$scene reconstruct_status=$recon_status time=$(date --iso-8601=seconds)" | tee -a "$scene_status" "$RUN_DIR/status.tsv"
done

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/status.tsv"
