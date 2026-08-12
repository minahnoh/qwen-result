#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_SOURCE="${1:-/home/elicer/qwen_defect/data/scenario2}"
cd "$PROJECT_DIR"

python3 -m pip install -r requirements.txt
python3 prepare_data.py --source "$DATA_SOURCE"
python3 train.py --data-root "$DATA_SOURCE" --output-dir /home/elicer/qwen_defect/result/scenario2-3 --epochs 3

