#!/usr/bin/env bash
# Mirror ComfyUI + the H3 DiT worker payload onto the peer Spark.
set -euo pipefail

PEER="${H3_SP_PEER_HOST:-spark-2}"
SRC="${COMFY_ROOT:-/home/xujie/workspace/ComfyUI}"
DST="$PEER:$SRC"

UNET="${H3_SP_UNET:-minimax_h3_ref2va_pruned_int8_convrot.safetensors}"
LORA="${H3_SP_LORA:-minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors}"

echo "syncing ComfyUI code + venv -> $DST"
rsync -a --info=progress2 \
  --exclude '/models/' \
  --exclude '/output/' \
  --exclude '/input/' \
  --exclude '/.git/' \
  --exclude '/__pycache__/' \
  --exclude '/custom_nodes/buqi-minimax-h3-multigpu/.git/' \
  "$SRC/" "$DST/"

echo "syncing DiT $UNET"
mkdir -p /tmp
ssh "$PEER" "mkdir -p '$SRC/models/diffusion_models' '$SRC/models/loras'"
rsync -a --info=progress2 \
  "$SRC/models/diffusion_models/$UNET" \
  "$DST/models/diffusion_models/"

echo "syncing LoRA $LORA"
rsync -a --info=progress2 \
  "$SRC/models/loras/$LORA" \
  "$DST/models/loras/"

echo "peer python:"
ssh "$PEER" "'$SRC/.venv/bin/python' -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'"

echo "done"
