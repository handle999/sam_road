#!/bin/bash
# Unified metric evaluation script
# Usage: bash eval.sh --dataset cityscale --dir save/xxx
#        bash eval.sh --dataset spacenet --dir save/xxx --workers 8
#        bash eval.sh --dataset spacenet --dir save/xxx --metric apls

python eval.py "$@"
