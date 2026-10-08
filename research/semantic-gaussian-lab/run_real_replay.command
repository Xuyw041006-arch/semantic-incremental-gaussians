#!/bin/zsh
cd "$(dirname "$0")"
LAB_PYTHON="/Users/xyw/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [[ ! -x "$LAB_PYTHON" ]]; then
  LAB_PYTHON="python3"
fi
"$LAB_PYTHON" -m sglab.replay --manifest ../real-experiment/data/tum-desk/panoptic.json --result ../real-experiment/results/cuda-replay-24000 --port 8766
