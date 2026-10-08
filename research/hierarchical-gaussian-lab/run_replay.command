#!/bin/zsh
set -eu
cd "${0:A:h}"
result_dir="../hierarchical-kitchen-results"
if [[ ! -f "$result_dir/viewer.json" ]]; then
  print '请将 hierarchical-kitchen-results.zip 解压到本目录的同级目录 hierarchical-kitchen-results。'
  exit 1
fi
runtime_python="/Users/xyw/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
if [[ ! -x "$runtime_python" ]]; then runtime_python="$(command -v python3)"; fi
if ! "$runtime_python" -c 'import numpy, PIL' 2>/dev/null; then
  print "需要 NumPy 和 Pillow，请执行：$runtime_python -m pip install numpy pillow"
  exit 1
fi
print '真实地图回放：http://127.0.0.1:8767/'
print '如果该服务已经运行，直接打开上面的地址。按 Ctrl+C 结束本次启动的服务。'
exec "$runtime_python" -m hglab.viewer --data "$result_dir" --port 8767
