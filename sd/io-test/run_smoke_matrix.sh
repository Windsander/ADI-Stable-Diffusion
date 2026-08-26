#!/bin/bash
# ADI smoke matrix runner (stage-3: per-scheduler golden/smoke regression)
#
# Usage:
#   bash sd/io-test/run_smoke_matrix.sh [adi-binary] [quick|full|turbo|sd35|flux|svd]
#
# Gate rules (fail fast, non-zero exit):
#   1. any "exception" in run log        -> FAIL (models silently return zeros otherwise)
#   2. missing / tiny output image       -> FAIL
#   3. flat output (pixel std < 5)       -> FAIL (zero-latent symptom)
# Visual quality is left to golden-image comparison (comparisons/ dir);
# this script guards the plumbing, not aesthetics.

set -u

ADI_BIN="${1:-cmake-build-debug-macos-arm64/bin/adi}"
MODE="${2:-quick}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$ROOT/sd/io-test"
LOG_DIR="$ROOT/sd/io-test/smoke-logs"
mkdir -p "$LOG_DIR"

FAILURES=0
RUNS=0

run_case() {
  local tag="$1"; shift
  local out="$OUT_DIR/output-smoke-$tag.png"
  local log="$LOG_DIR/$tag.log"
  RUNS=$((RUNS + 1))
  echo "== [$RUNS] $tag"
  "$ADI_BIN" -p "A cat in the water at sunset" -m txt2img -o "$out" "$@" \
      --token-idx-num 49408 --token-length 77 --token-border 1.0 --gain 1.1 \
      --strength 0.0 > "$log" 2>&1

  local exc_count
  exc_count=$(grep -icE "exception" "$log" || true)
  if [ "$exc_count" != "0" ]; then
    echo "   FAIL: $exc_count ORT exceptions (see $log)"; FAILURES=$((FAILURES + 1)); return
  fi
  if [ ! -f "$out" ] || [ "$(stat -f%z "$out")" -lt 50000 ]; then
    echo "   FAIL: output missing or suspiciously small"; FAILURES=$((FAILURES + 1)); return
  fi
  local flat="OK"
  if python3 -c "import PIL" 2>/dev/null; then
    flat=$(python3 - "$out" <<'EOF'
import sys
from PIL import Image
import numpy as np
im = np.array(Image.open(sys.argv[1]).convert('RGB'), dtype=float)
print("FLAT" if im.std() < 5 else "OK")
EOF
)
  else
    echo "   (warn: python3+PIL unavailable, flat check skipped)"
  fi
  if [ "$flat" != "OK" ]; then
    echo "   FAIL: output is flat (std < 5) — zero-latent symptom"; FAILURES=$((FAILURES + 1)); return
  fi
  echo "   ok"
}

model_args() {
  local m="$ROOT/sd/sd-base-model/$1"
  echo "--clip $m/text_encoder/model.onnx --unet $m/unet/model.onnx"
  echo "--vae-encoder $m/vae_encoder/model.onnx --vae-decoder $m/vae_decoder/model.onnx"
  echo "--merges $m/tokenizer/merges.txt --dict $m/tokenizer/vocab.json"
}

# img2vid variant: output is a frame sequence <base>_NNNN.png
run_video_case() {
  local tag="$1"; local frames="$2"; shift 2
  local out="$OUT_DIR/output-smoke-$tag.png"
  local log="$LOG_DIR/$tag.log"
  RUNS=$((RUNS + 1))
  echo "== [$RUNS] $tag (${frames} frames)"
  "$ADI_BIN" -m img2vid -i "$OUT_DIR/input.png" -o "$out" --frames "$frames" "$@" \
      > "$log" 2>&1

  local exc_count
  exc_count=$(grep -icE "exception" "$log" || true)
  if [ "$exc_count" != "0" ]; then
    echo "   FAIL: $exc_count ORT exceptions (see $log)"; FAILURES=$((FAILURES + 1)); return
  fi
  local base="${out%.png}"
  local last_frame
  last_frame=$(printf "%s_%04d.png" "$base" $((frames - 1)))
  if [ ! -f "${base}_0000.png" ] || [ ! -f "$last_frame" ]; then
    echo "   FAIL: frame sequence incomplete (missing first/last frame)"; FAILURES=$((FAILURES + 1)); return
  fi
  local actual
  actual=$(ls "${base}_"????.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$actual" != "$frames" ]; then
    echo "   FAIL: expected $frames frames, got $actual"; FAILURES=$((FAILURES + 1)); return
  fi
  local flat="OK"
  if python3 -c "import PIL" 2>/dev/null; then
    flat=$(python3 - "${base}_0000.png" <<'EOF'
import sys
from PIL import Image
import numpy as np
im = np.array(Image.open(sys.argv[1]).convert('RGB'), dtype=float)
print("FLAT" if im.std() < 5 else "OK")
EOF
)
  fi
  if [ "$flat" != "OK" ]; then
    echo "   FAIL: first frame is flat (std < 5) — zero-latent symptom"; FAILURES=$((FAILURES + 1)); return
  fi
  echo "   ok"
}

BASE="--beta-start 0.00085 --beta-end 0.012 --beta scaled_linear --alpha cos --tokenizer bpe --train-steps 1000"

# ---------- sd-turbo (dims 1024, guidance 1.0) ----------
# 默认矩阵仅在 quick/full/turbo 模式下运行；sd35/flux/svd 等单模型模式直接跳转到各自段落
if [ "$MODE" == "quick" ] || [ "$MODE" == "full" ] || [ "$MODE" == "turbo" ]; then
TURBO_SCHEDS="euler_a unipc dpm_m dpm_sde dpm_s pndm ipndm deis_m"
for s in $TURBO_SCHEDS; do
  run_case "turbo-$s-s4" $(model_args onnx-sd-turbo) \
    -w 512 -h 512 -c 3 --seed 15.0 --dims 1024 $BASE \
    --scheduler $s --predictor epsilon --guidance 1.0 --steps 4
done
for s in euler_a unipc pndm ipndm deis_m; do
  run_case "turbo-$s-karras-s4" $(model_args onnx-sd-turbo) \
    -w 512 -h 512 -c 3 --seed 15.0 --dims 1024 $BASE \
    --scheduler $s --sigma karras --predictor epsilon --guidance 1.0 --steps 4
done
fi

if [ "$MODE" == "turbo" ]; then
  echo "============================================"
  echo "smoke matrix done (turbo only): $RUNS runs, $FAILURES failures"
  [ "$FAILURES" == "0" ]
  exit $?
fi

# ---------- sd3.5-large-turbo (triple encoder, MMDiT, flow_euler, 1024px) ----------
if [ "$MODE" == "sd35" ]; then
  # 支持 fp32 / fp16 雙目录，SD35_MODEL_DIR 環境變量切換。
  # fp16 由 sd/tools/onnx_fp16_convert.py（ORT transformers float16 轉換器）重建，
  # 2026-08-25 全量 verify OK + 1024px 冒煙通過（std=47.39 vs fp32 47.53）。
  # 默認 fp32（高內存機器直連）；低內存機器設 SD35_MODEL_DIR=...-fp16。
  SD35="${SD35_MODEL_DIR:-$ROOT/sd/sd-base-model/onnx-sd35-turbo}"
  # -c は IO 画像チャンネル数（常に 3）。latent 16ch は UNet 4-input 検出で自動適用。
  # --decoding は VAE scaling_factor 自体（decode: latents/scale + shift）→ SD3.5 = 1.5305。
  for steps in 4 8; do
    run_case "sd35-flow_euler-s${steps}" \
      --clip  $SD35/text_encoder/model.onnx \
      --clip2 $SD35/text_encoder_2/model.onnx \
      --clip3 $SD35/text_encoder_3/model.onnx \
      --unet  $SD35/transformer/model.onnx \
      --vae-encoder $SD35/vae_encoder/model.onnx \
      --vae-decoder $SD35/vae_decoder/model.onnx \
      --merges $SD35/tokenizer/merges.txt \
      --dict  $SD35/tokenizer/vocab.json \
      --sp-model $SD35/tokenizer_3/spiece.model \
      -w 1024 -h 1024 -c 3 --seed 15.0 --dims 768 \
      --beta scaled_linear --scheduler flow_euler --shift 3.0 \
      --predictor epsilon --tokenizer bpe \
      --token-idx-num 49408 --token-length 77 \
      --decoding 1.5305 --decode-shift 0.0609 \
      --guidance 1.0 --steps $steps
  done
  echo "============================================"
  echo "smoke matrix done (sd35): $RUNS runs, $FAILURES failures"
  [ "$FAILURES" == "0" ]
  exit $?
fi

# ---------- FLUX.1-schnell (dual encoder, packed MMDiT, flow_euler, 1024px) ----------
if [ "$MODE" == "flux" ]; then
  # schnell は guidance 蒸留済み：guidance=1.0 / shift=1.0（scheduler_config 準拠、
  # use_dynamic_shifting=false）。timestep は C++ 側で自動 /1000（diffusers 準拠）。
  # VAE: scaling_factor=0.3611, shift_factor=0.1159。
  FLUX="${FLUX_MODEL_DIR:-$ROOT/sd/sd-base-model/onnx-flux-schnell}"
  for steps in 4; do
    run_case "flux-flow_euler-s${steps}" \
      --clip  $FLUX/text_encoder/model.onnx \
      --clip3 $FLUX/text_encoder_2/model.onnx \
      --unet  $FLUX/transformer/model.onnx \
      --vae-encoder $FLUX/vae_encoder/model.onnx \
      --vae-decoder $FLUX/vae_decoder/model.onnx \
      --merges $FLUX/tokenizer/merges.txt \
      --dict  $FLUX/tokenizer/vocab.json \
      --sp-model $FLUX/tokenizer_2/spiece.model \
      -w 1024 -h 1024 -c 3 --seed 15.0 --dims 768 \
      --beta scaled_linear --scheduler flow_euler --shift 1.0 \
      --predictor epsilon --tokenizer bpe \
      --token-idx-num 49408 --token-length 77 \
      --decoding 0.3611 --decode-shift 0.1159 \
      --guidance 1.0 --steps $steps
  done
  echo "============================================"
  echo "smoke matrix done (flux): $RUNS runs, $FAILURES failures"
  [ "$FAILURES" == "0" ]
  exit $?
fi

# ---------- SVD-XT-1.1 (img2vid: CLIP vision + spatio-temporal UNet, 14f 576x1024) ----------
if [ "$MODE" == "svd" ]; then
  # euler_svd: karras sigma ramp 700 -> 0.002 (scheduler_config.json 準拠)、
  # continuous timestep (0.25*lnσ)、v_prediction はクラス内で強制。
  # --decoding は VAE scaling_factor=0.18215（decode: latents/scale; encode 側は
  # SVD 仕様によりスケール無し・C++ 側で encoder scale=1.0 固定）。
  # fps=7 は C++ 側で fps-1=6 として added_time_ids に入る（diffusers 準拠）。
  SVD="${SVD_MODEL_DIR:-$ROOT/sd/sd-base-model/onnx-svd-xt}"
  for steps in 4; do
    run_video_case "svd-euler_svd-s${steps}" 14 \
      --image-encoder $SVD/image_encoder/model.onnx \
      --unet  $SVD/unet/model.onnx \
      --vae-encoder $SVD/vae_encoder/model.onnx \
      --vae-decoder $SVD/vae_decoder/model.onnx \
      -w 1024 -h 576 -c 3 --seed 15.0 \
      --scheduler euler_svd --predictor v_prediction \
      --fps 7 --motion-bucket 127 --noise-aug 0.02 \
      --decoding 0.18215 --guidance 3.0 --steps $steps
  done
  echo "============================================"
  echo "smoke matrix done (svd): $RUNS runs, $FAILURES failures"
  [ "$FAILURES" == "0" ]
  exit $?
fi

# ---------- sd v1.5 (dims 768, guidance 7.5, 20 steps) ----------
if [ "$MODE" == "quick" ] || [ "$MODE" == "full" ]; then
for s in euler_a unipc dpm_m pndm ipndm deis_m; do
  run_case "v15-$s-s20" $(model_args onnx-sd-v15) \
    -w 512 -h 512 -c 3 --seed 15.0 --dims 768 $BASE \
    --scheduler $s --predictor epsilon --guidance 7.5 --steps 20
done
fi

if [ "$MODE" == "full" ]; then
  # ---------- sd v2.1 768px (dims 1024, v_prediction) ----------
  for s in euler_a dpm_m; do
    run_case "v21-$s-s20" $(model_args onnx-sd-v21-768) \
      -w 768 -h 768 -c 3 --seed 15.0 --dims 1024 $BASE \
      --scheduler $s --predictor v_prediction --guidance 7.5 --steps 20
  done

  # ---------- sdxl-turbo (dual encoder, decoding 0.13025) ----------
  SDXL="$ROOT/sd/sd-base-model/onnx-sdxl-turbo"
  run_case "sdxl-euler_a-s4" \
    --clip $SDXL/text_encoder/model.onnx --clip2 $SDXL/text_encoder_2/model.onnx \
    --unet $SDXL/unet/model.onnx \
    --vae-encoder $SDXL/vae_encoder/model.onnx --vae-decoder $SDXL/vae_decoder/model.onnx \
    --merges $SDXL/tokenizer/merges.txt --dict $SDXL/tokenizer/vocab.json \
    -w 512 -h 512 -c 3 --seed 15.0 --dims 768 $BASE \
    --scheduler euler_a --predictor epsilon --decoding 0.13025 --guidance 1.0 --steps 4
fi

echo "============================================"
echo "smoke matrix done: $RUNS runs, $FAILURES failures"
[ "$FAILURES" == "0" ]
