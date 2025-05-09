#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="./data"
KAGGLE_SCRIPT="kaggle datasets download" 
ELSA_D3_SCRIPT="$DATA_ROOT/scripts/download_elsa_d3.py"
ELSA_D3_N_FILES=10

mkdir -p "$DATA_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not in PATH."
  exit 1
fi


uv run $KAGGLE_SCRIPT -d selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models -p $DATA_ROOT/datasets/sfhq_t2i --unzip
uv run $KAGGLE_SCRIPT -d alessandrasala79/ai-vs-human-generated-dataset -p $DATA_ROOT/datasets/ai_vs_human_generated --unzip
uv run $KAGGLE_SCRIPT -d mohannadaymansalah/stable-diffusion-dataaaaaaaaa -p $DATA_ROOT/datasets/sd_faces --unzip
uv run $KAGGLE_SCRIPT -d programmerrdai/open-images-v6 -p $DATA_ROOT/datasets/openimages_v6 --unzip
uv run $ELSA_D3_SCRIPT $ELSA_D3_N_FILES --dest $DATA_ROOT/datasets/slice_elsa_d3
