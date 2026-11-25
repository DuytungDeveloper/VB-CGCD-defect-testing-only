#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
FT_EPOCHS="${FT_EPOCHS:-20}"
DYDM_EPOCHS="${DYDM_EPOCHS:-12}"
FTS_CSV="${FTS_CSV:-1,2}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a9_heldout_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a9_heldout_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/gc10_a9_heldout_telemetry.json}"

TOTAL_T0=$(date +%s)
trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc, total_sec=$(($(date +%s) - TOTAL_T0))) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== START GC10 STEP A9 HELD-OUT CONFIRMATION (FTS=${FTS_CSV}, SEEDS=${SEEDS_CSV}, A8 CHAMPION) at $(date) ==="

python - <<PY
import json, os, platform, subprocess
from pathlib import Path

cpu_model = platform.processor() or "unknown"
try:
    for ln in Path("/proc/cpuinfo").read_text().splitlines():
        if "model name" in ln:
            cpu_model = ln.split(":", 1)[1].strip()
            break
except Exception:
    pass

ram_gb = 0.0
try:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemTotal:"):
            ram_gb = round(int(ln.split()[1]) / 1024 / 1024, 2)
            break
except Exception:
    pass

gpu_info = "NO_GPU"
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True
    ).strip()
except Exception:
    pass

import torch, jax
meta = {
    "model": "GC10-DET A8 Champion Held-out (DINOv3 + Auto-ROI Fusion + Imbalanced-DYDM)",
    "cpu_model": cpu_model,
    "cpu_cores": os.cpu_count(),
    "ram_total_gb": ram_gb,
    "gpu": gpu_info,
    "torch_version": torch.__version__,
    "torch_cuda": torch.cuda.is_available(),
    "jax_version": jax.__version__,
    "jax_devices": [str(d) for d in jax.devices()],
    "stages": {},
}
p = Path("$META_JSON")
if p.exists():
    try:
        old = json.loads(p.read_text())
        meta["stages"] = old.get("stages", {})
    except Exception:
        pass
p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"HARDWARE | CPU: {cpu_model} ({os.cpu_count()} cores) | RAM: {ram_gb} GB | GPU: {gpu_info}")
print(f"FRAMEWORKS | Torch {torch.__version__} (CUDA={torch.cuda.is_available()}) | JAX {jax.__version__} ({jax.devices()})")
PY

record_stage() {
  local key="$1" sec="$2" hw_file="$3"
  python - <<PY
import json
from pathlib import Path
p = Path("$META_JSON")
d = json.loads(p.read_text()) if p.exists() else {"stages": {}}
hw = json.loads(Path("$hw_file").read_text()) if Path("$hw_file").exists() else {}
hw["sec"] = int("$sec")
d.setdefault("stages", {})["$key"] = hw
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
  rm -f "$hw_file"
}

link_gc10_dataset() {
  mkdir -p datasets
  if [ -d "datasets/GC10-DET/1" ] && [ ! -L "datasets/GC10-DET" ]; then
    return 0
  fi
  if [ -d "/content/GC10-DET/1" ]; then
    rm -f datasets/GC10-DET
    ln -sfn "/content/GC10-DET" datasets/GC10-DET
    return 0
  fi
  for cand in "/content/drive/MyDrive/Datasets/GC10-DET" "$WORK_DIR/GC10-DET" "$WORK_DIR/Datasets/GC10-DET"; do
    if [ -d "$cand/1" ]; then
      echo ">>> Copy GC10-DET từ Drive sang /content/GC10-DET..."
      cp -r "$cand" /content/GC10-DET
      rm -f datasets/GC10-DET
      ln -sfn "/content/GC10-DET" datasets/GC10-DET
      return 0
    fi
  done
  echo "ERROR: Không tìm thấy GC10-DET tại /content/drive/MyDrive/Datasets/GC10-DET" >&2
  return 1
}

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for pat in "*__colab_gc10_dinov3_ep${FT_EPOCHS}_r*" "*__colab_gc10_d3_autoroi_ep${FT_EPOCHS}_r*"; do
    for d in "$DRIVE_REPO"/features/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "features/$b" ] || { echo ">>> Đồng bộ features từ Drive: $b"; cp -r "$d" "features/$b"; }
    done
    for d in "$DRIVE_REPO"/runs/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "runs/$b" ] || { echo ">>> Đồng bộ run từ Drive: $b"; cp -r "$d" "runs/$b"; }
    done
  done
}

backup_r_to_drive() {
  local r="$1"
  mkdir -p "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for pat in "*__colab_gc10_dinov3_ep${FT_EPOCHS}_r${r}" "*__colab_gc10_d3_autoroi_ep${FT_EPOCHS}_r${r}"; do
    for d in features/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
    done
  done
  for d in runs/*__colab_gc10_d3_autoroi_ep"${FT_EPOCHS}"_r"${r}"_s*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ -d "$d/snap" ]; then
      rm -rf "$DRIVE_REPO/runs/$b"
      cp -r "$d" "$DRIVE_REPO/runs/$b"
    fi
  done
}

run_one_ft_seed() {
  local r="$1"
  local p1_tag="colab_gc10_dinov3_ep${FT_EPOCHS}_r${r}"
  local roi_tag="colab_gc10_d3_autoroi_ep${FT_EPOCHS}_r${r}"
  local p1_root="" roi_root="" roi_feat=""

  # 1. Phase 1 DINOv3 fine-tune
  p1_root=$(ls -d features/*__"${p1_tag}" 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "$p1_root" ] && [ -f "$p1_root/dinov3-vitb16-sl/features-dinov3-vitb16-sl.npy" ] && [ -f "$p1_root/ft_ckpt.pt" ]; then
    echo ">>> Đã có Phase 1 DINOv3 r${r} tại ${p1_root}, bỏ qua."
  else
    echo ">>> [r${r}] Chạy Phase 1 DINOv3 (FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${r}) lúc $(date)..."
    local t0 hw_tmp mon_pid
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a9_p1_r${r}.json"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
    FT_EPOCHS="$FT_EPOCHS" FT_SEED="$r" python scripts/exp.py run "$p1_tag" --inputs splits/gc10_split.json -- \
      bash scripts/run_phase1_neu.sh --split splits/gc10_split.json --resize full --model dinov3_vitb16
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "p1_r${r}" "$(($(date +%s) - t0))" "$hw_tmp"
    p1_root=$(ls -d features/*__"${p1_tag}" | sort | tail -n 1)
    backup_r_to_drive "$r"
  fi

  # 2. DINOv3 Auto-ROI Fusion
  roi_root=$(ls -d features/*__"${roi_tag}" 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "$roi_root" ] && [ -f "$roi_root/dinov3-vitb16-sl/features-dinov3-vitb16-sl.npy" ] && [ -f "$roi_root/ft_ckpt.pt" ]; then
    echo ">>> Đã có Auto-ROI Fusion r${r} tại ${roi_root}, bỏ qua."
  else
    local ts
    ts=$(date -u +%Y%m%d_%H%M%S)
    roi_root="features/${ts}__${roi_tag}"
    echo ">>> [r${r}] Trích xuất DINOv3 Auto-ROI Fusion -> ${roi_root} lúc $(date)..."
    local t0 hw_tmp mon_pid
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a9_roi_r${r}.json"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
    python colab/extract_gc10_a7a_dinov3_roi.py \
      --a2b-feat-root "$p1_root" \
      --out-root "$roi_root" \
      --split splits/gc10_split.json \
      --alpha 0.5 \
      --loc-size 384 \
      --batch-size 32
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "roi_r${r}" "$(($(date +%s) - t0))" "$hw_tmp"
    backup_r_to_drive "$r"
  fi
  roi_feat="${roi_root}/dinov3-vitb16-sl"

  # 3. Phase 2 sessions (s0..s4)
  IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
  for s in "${SEEDS[@]}"; do
    local stag="${roi_tag}_s${s}"
    if ls runs/*__"${stag}"/snap/S1_14.pkl >/dev/null 2>&1; then
      echo ">>> Đã hoàn thành Phase 2 ${stag}, bỏ qua."
      continue
    fi
    local t0 hw_tmp mon_pid prev_dir
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a9_p2_r${r}_s${s}.json"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
    prev_dir=$(ls -d runs/*__"${stag}" 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$prev_dir" ] && [ -d "$prev_dir/snap" ]; then
      python scripts/exp.py run "$stag" --inputs "$roi_feat" -- \
        python lab/sessions.py run --features "$roi_feat" --seed "$s" --dataset gc10 --resume "$prev_dir"
      rm -rf "$prev_dir"
    else
      [ -n "$prev_dir" ] && rm -rf "$prev_dir"
      python scripts/exp.py run "$stag" --inputs "$roi_feat" -- \
        python lab/sessions.py run --features "$roi_feat" --seed "$s" --dataset gc10
    fi
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "p2_r${r}_s${s}" "$(($(date +%s) - t0))" "$hw_tmp"
    backup_r_to_drive "$r"
  done

  # 4. Imbalanced-DYDM (A8) for r
  local r_json="$WORK_DIR/gc10_a9_r${r}_summary.json"
  for s in "${SEEDS[@]}"; do
    if python -c "import json; d=json.load(open('$r_json')); exit(0 if 's${s}' in d.get('per_seed',{}) else 1)" 2>/dev/null; then
      echo ">>> Đã có kết quả A8 ImbDYDM r${r}_s${s}, bỏ qua."
      continue
    fi
    local t0 hw_tmp mon_pid
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a9_dydm_r${r}_s${s}.json"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
    echo ">>> [r${r}] Chạy A8 Imbalanced-DYDM cho seed s${s} ($(date))..."
    python colab/run_gc10_a8_autoroi_dydm.py \
      --prefix "$roi_tag" \
      --feat-dir "$roi_feat" \
      --ft-ckpt "${roi_root}/ft_ckpt.pt" \
      --seeds "$s" \
      --epochs "$DYDM_EPOCHS" \
      --out-json "$r_json" \
      --meta-json "$META_JSON"
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "dydm_r${r}_s${s}" "$(($(date +%s) - t0))" "$hw_tmp"
  done
}

aggregate_all() {
  {
    python - <<PY
import json
from pathlib import Path
import numpy as np

work = Path("$WORK_DIR")
meta = json.loads(Path("$META_JSON").read_text()) if Path("$META_JSON").exists() else {}
stages = meta.get("stages", {})
cpu_pk = max((v.get("cpu_peak_pct", 0) for v in stages.values()), default=0)
ram_pk = max((v.get("ram_peak_gb", 0) for v in stages.values()), default=0)
gpu_pk = max((v.get("gpu_util_peak_pct", 0) for v in stages.values()), default=0)
vram_pk = max((v.get("gpu_vram_peak_mb", 0) for v in stages.values()), default=0)
tot_sec = sum(v.get("sec", 0) for v in stages.values())

print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC A9 - HELD-OUT r1, r2) ===")
print(f"CPU          : {meta.get('cpu_model')} ({meta.get('cpu_cores')} cores) | Peak CPU: {cpu_pk}%")
print(f"RAM          : Peak {ram_pk} GB / {meta.get('ram_total_gb')} GB")
print(f"GPU          : {meta.get('gpu')} | Peak GPU Util: {gpu_pk}% | Peak GPU VRAM: {vram_pk} MB")
print(f"PyTorch      : {meta.get('torch_version')} (CUDA={meta.get('torch_cuda')}) | JAX {meta.get('jax_version')} ({meta.get('jax_devices')})")
print(f"Tổng thời gian các công đoạn mới chạy: {tot_sec} s ({tot_sec/60:.1f} phút)\n")

# Load r0 (from A8), r1, r2
sources = {
    "r0": [work / "gc10_a8_autoroi_dydm_summary.json", Path("runs/gc10_a8_autoroi_dydm_summary.json")],
    "r1": [work / "gc10_a9_r1_summary.json"],
    "r2": [work / "gc10_a9_r2_summary.json"],
}
loaded = {}
novel_names = []
for rk, cands in sources.items():
    for cp in cands:
        if cp.exists():
            d = json.loads(cp.read_text())
            loaded[rk] = d
            if not novel_names and d.get("novel_classes"):
                novel_names = d["novel_classes"]
            break

arms = ("N", "NL", "A8_Imb_head", "A8_Imb_km", "A8_Imb_blend50", "A8_Imb_blend70")
grand = {}
per_ft = {}
for rk, d in sorted(loaded.items()):
    per_ft[rk] = d.get("arms", {})

for arm in arms:
    all_vals = []
    for rk, d in sorted(loaded.items()):
        for sk, sv in sorted(d.get("per_seed", {}).items()):
            if arm in sv:
                all_vals.append(sv[arm])
    if not all_vals:
        continue
    pc_mean = {}
    for idx_c, cname in enumerate(novel_names):
        cv = [v.get("per_class_novel", {}).get(str(idx_c), v.get("per_class_novel", {}).get(idx_c, 0.0)) for v in all_vals]
        pc_mean[cname] = float(np.mean(cv))
    grand[arm] = {
        "Old_mean": float(np.mean([v["Old"] for v in all_vals])),
        "Old_std": float(np.std([v["Old"] for v in all_vals])),
        "New_mean": float(np.mean([v.get("New", 0.0) for v in all_vals])),
        "New_std": float(np.std([v.get("New", 0.0) for v in all_vals])),
        "bal_mean": float(np.mean([v["bal"] for v in all_vals])),
        "bal_std": float(np.std([v["bal"] for v in all_vals])),
        "c09_mean": float(np.mean([v["c09"] for v in all_vals])),
        "All_mean": float(np.mean([v["All"] for v in all_vals])),
        "All_std": float(np.std([v["All"] for v in all_vals])),
        "per_class_novel": pc_mean,
        "n_chains": len(all_vals),
    }

out_payload = {"fts_loaded": sorted(loaded.keys()), "novel_classes": novel_names, "grand_15chains": grand, "per_ft": per_ft, "telemetry": meta}
Path("$SUMMARY_JSON").write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))

print(f"=== BẢNG TỔNG HỢP CHỐT THỐNG KÊ ({', '.join(sorted(loaded.keys()))} — TỔNG {grand.get('A8_Imb_head', {}).get('n_chains', 0)} CHUỖI ĐỘC LẬP) ===")
print("Ghi chú:")
print("  - Lớp cũ (Old %)     : Độ chính xác nhận diện 5 lớp lỗi đã học ở S0 (càng cao càng ít quên).")
print("  - Lớp mới (New %)    : % tổng số ảnh thuộc 5 lớp lỗi mới được khám phá & phân loại đúng.")
print("  - Cân bằng mới (bal) : Trung bình độ chính xác từng lớp mới (0..1, công bằng cho lớp lỗi hiếm).")
print("  - Gom cụm (c09 %)    : Độ thuần khi tự gom cụm 5 lớp mới chưa có nhãn.")
print("  - Tổng thể (All %)   : Độ chính xác chung trên toàn bộ 10 lớp lỗi thép.")
print()
print(f"{'Phương pháp':<16} {'Lớp cũ(Old%)':>13} {'Lớp mới(New%)':>14} {'Cân bằng(bal)':>18} {'Gom cụm(c09%)':>14} {'Tổng(All%±std)':>17} {'N':>4}")
print("-" * 102)
for arm in arms:
    if arm not in grand:
        continue
    r = grand[arm]
    print(f"{arm:<16} {r['Old_mean']:12.2f}% {r['New_mean']:13.2f}% {r['bal_mean']:7.4f} ± {r['bal_std']:6.4f} {r['c09_mean']:13.2f}% {r['All_mean']:7.2f} ± {r['All_std']:5.2f}% {r['n_chains']:4d}")

print("\n--- ĐỐI CHIẾU TỪNG LẦN FINE-TUNE ĐỘC LẬP (A8_Imb_head vs N) ---")
for rk in sorted(per_ft.keys()):
    a8 = per_ft[rk].get("A8_Imb_head", {})
    na = per_ft[rk].get("N", {})
    if a8:
        print(f"  [{rk}] N: All={na.get('All_mean',0):.2f}%, bal={na.get('bal_mean',0):.4f} | A8_Imb_head: Old={a8.get('Old_mean',0):.2f}%, New={a8.get('New_mean',0):.2f}%, bal={a8.get('bal_mean',0):.4f}±{a8.get('bal_std',0):.4f}, c09={a8.get('c09_mean',0):.2f}%, All={a8.get('All_mean',0):.2f}%")

if novel_names:
    print("\n--- CHI TIẾT ĐỘ CHÍNH XÁC TỪNG LỚP LỖI MỚI (%, TRUNG BÌNH TẤT CẢ CHUỖI) ---")
    hdr = f"{'Phương pháp':<16} " + " ".join(f"{c[:12]:>12}" for c in novel_names)
    print(hdr)
    print("-" * len(hdr))
    for arm in arms:
        if arm not in grand:
            continue
        pc = grand[arm].get("per_class_novel", {})
        print(f"{arm:<16} " + " ".join(f"{pc.get(c, 0.0):11.1f}%" for c in novel_names))
PY
  } | tee "$SUMMARY_OUT"
}

link_gc10_dataset
sync_from_drive

IFS=',' read -ra FTS <<< "$FTS_CSV"
case "$STAGE" in
  auto)
    for r in "${FTS[@]}"; do
      run_one_ft_seed "$r"
    done
    aggregate_all
    ;;
  eval)
    aggregate_all
    ;;
  *)
    echo "Stage không hợp lệ: $STAGE (chọn: auto | eval)" >&2
    exit 2
    ;;
esac
