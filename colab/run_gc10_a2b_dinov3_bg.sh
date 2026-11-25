#!/usr/bin/env bash
set -euo pipefail

# Step A2b: DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m) on GC10-DET (FT_EPOCHS=20, r0, s0..s4)
# Requires HF_TOKEN exported in environment (with approved access to facebook/dinov3-vitb16-pretrain-lvd1689m).
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Chưa thiết lập biến môi trường HF_TOKEN! Hãy set HF_TOKEN (hoặc bật Colab Secret HF_TOKEN) trước khi chạy." >&2
  exit 2
fi

export MODEL="dinov3_vitb16"
export FT_EPOCHS="${FT_EPOCHS:-20}"
export PREFIX="colab_gc10_dinov3_ep${FT_EPOCHS}"
export SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a2b_dinov3_summary.txt}"
export SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a2b_dinov3_summary.json}"
export META_JSON="${META_JSON:-$WORK_DIR/gc10_a2b_dinov3_telemetry.json}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

exec bash colab/run_gc10_a2_bg.sh "${1:-auto}"
