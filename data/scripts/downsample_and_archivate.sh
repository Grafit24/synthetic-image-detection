#!/usr/bin/env bash
set -euo pipefail

# Default values
QUALITY=85
SIZE="768x768"
WORKERS=16

# Parse options
while getopts ":q:s:w:" opt; do
  case $opt in
    q) QUALITY="$OPTARG" ;;
    s) SIZE="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
    :) echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
  esac
done
shift $((OPTIND -1))

# Positional DATASETS_DIR or default
DATASETS_DIR="${1:-data/datasets}"

# Directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for dataset_path in "${DATASETS_DIR}"/*; do
  [ -d "$dataset_path" ] || continue
  dataset_name="$(basename "$dataset_path")"
  echo "→ Processing dataset: $dataset_name"

  uv run "${SCRIPT_DIR}/downsample_datasets.py" \
    -d "$dataset_path" \
    -w "$WORKERS" \
    -q "$QUALITY" \
    -s "$SIZE"

  zip_file="${DATASETS_DIR}/${dataset_name}_lowres.zip"
  echo "   Zipping results into $zip_file"

  # find all lowres directories anywhere under the dataset
  mapfile -t LOWRES_DIRS < <(find "$dataset_path" -type d -name "lowres")

  if [ ${#LOWRES_DIRS[@]} -gt 0 ]; then
    zip -r "$zip_file" \
      "$dataset_path/processed.csv" \
      "${LOWRES_DIRS[@]}"
  else
    echo "   Warning: no 'lowres' directories found, zipping only processed.csv"
    zip -j "$zip_file" "$dataset_path/processed.csv"
  fi

  echo "✔ Archive $zip_file complete"
done
