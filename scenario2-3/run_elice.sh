#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_SOURCE="${1:-/home/elicer/qwen_defect/elice_scenario2_2}"
cd "$PROJECT_DIR"

python3 -m pip install -r requirements.txt
python3 prepare_data.py --source "$DATA_SOURCE" --destination data
python3 train.py --data-root data --output-dir outputs/scenario2-3 --epochs 3
