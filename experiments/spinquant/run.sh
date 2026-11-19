#!/usr/bin/env bash
# SPDX-License-Identifier: CC-BY-NC-4.0
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo 'Usage: bash run.sh SPINQUANT_CHECKOUT MODEL_DIR ROTATION_FILE' >&2
  exit 2
fi

SPINQUANT_DIR=$1
MODEL=$2
ROTATION=$3
W_BITS=${W_BITS:-2}
PAC_RANK=${PAC_RANK:-29}
OGC_RANK=${OGC_RANK:-3}

case "$W_BITS" in
  2|3) ;;
  *) echo 'W_BITS must be 2 or 3.' >&2; exit 2 ;;
esac
if [[ ! "$PAC_RANK" =~ ^[0-9]+$ || ! "$OGC_RANK" =~ ^[0-9]+$ ]]; then
  echo 'PAC_RANK and OGC_RANK must be nonnegative integers.' >&2
  exit 2
fi
if [[ ! -f "$SPINQUANT_DIR/eval_utils/rocob_utils.py" ]]; then
  echo 'Apply rocob.patch to the SpinQuant checkout first.' >&2
  exit 2
fi
if [[ ! -d "$MODEL" || ! -f "$ROTATION" ]]; then
  echo 'Supply a local model directory and an optimized rotation file.' >&2
  exit 2
fi

[[ "$MODEL" == /* ]] || MODEL="$PWD/$MODEL"
[[ "$ROTATION" == /* ]] || ROTATION="$PWD/$ROTATION"
cd -- "$SPINQUANT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
printf 'SpinQuant W%sA16KV16, PAC_RANK=%s, OGC_RANK=%s\n' "$W_BITS" "$PAC_RANK" "$OGC_RANK" >&2
exec torchrun --standalone --nnodes=1 --nproc_per_node=1 ptq.py \
  --input_model "$MODEL" \
  --do_train False --do_eval True --per_device_eval_batch_size 1 \
  --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
  --w_bits "$W_BITS" --a_bits 16 --k_bits 16 --v_bits 16 \
  --w_groupsize -1 --no-act_order --w_asym --w_clip --percdamp 0.01 \
  --seed 0 --nsamples 128 --gptq_batch_size 64 \
  --rotate --optimized_rotation_path "$ROTATION" \
  --pac_rank "$PAC_RANK" --ogc_rank "$OGC_RANK"
