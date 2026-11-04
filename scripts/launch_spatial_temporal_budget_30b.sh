#!/usr/bin/env bash
# One process, sharded across free GPUs. Does not write the 8B experiment.
set -u
cd "$(dirname "$0")/.."
PY=/home/vailab02/anaconda3/envs/vlm_experiment_env/bin/python
LOG=experiments/spatial_temporal_budget_40960_30b/logs
mkdir -p "$LOG"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "==== download $(date -Is) ===="
$PY -u - << 'PY'
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi

repo = "Qwen/Qwen3-VL-30B-A3B-Instruct"
rev = "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
local = Path("/data1/vailab02_dir/vlm_models/Qwen3-VL-30B-A3B-Instruct")
api = HfApi()
files = [e for e in api.list_repo_tree(repo, revision=rev, recursive=True) if getattr(e, "size", None)]
todo = []
for e in files:
    dest = local / e.path
    if dest.is_file() and dest.stat().st_size == e.size:
        print(f"SKIP {e.path}", flush=True)
        continue
    todo.append(e)

def fetch(e):
    dest = local / e.path
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{e.path}"
    cmd = ["curl", "-L", "--retry", "12", "--retry-all-errors", "--retry-delay", "2", "-C", "-", "--fail", "-o", str(dest), url]
    print(f"GET {e.path}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    got = dest.stat().st_size if dest.is_file() else -1
    if proc.returncode != 0 or got != e.size:
        print(f"FAIL {e.path} rc={proc.returncode} got={got} expected={e.size}", flush=True)
        return False
    print(f"OK {e.path}", flush=True)
    return True

print(f"TODO {len(todo)}", flush=True)
ok = True
with ThreadPoolExecutor(max_workers=4) as pool:
    for fut in as_completed([pool.submit(fetch, e) for e in todo]):
        ok = fut.result() and ok
print("DOWNLOAD_DONE" if ok else "DOWNLOAD_BAD", flush=True)
raise SystemExit(0 if ok else 1)
PY
dl_rc=$?
echo "download_exit=$dl_rc"
if [ "$dl_rc" -ne 0 ]; then
  echo "download failed" >&2
  exit 1
fi

echo "==== gpus $(date -Is) ===="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>44000 {print $1}')
echo "free_gpus=${FREE[*]:-none}"
if [ "${#FREE[@]}" -lt 2 ]; then
  echo "need at least 2 free 48GB GPUs for BF16" >&2
  exit 2
fi

run_with() {
  local spec="$1"
  shift
  echo "CUDA_VISIBLE_DEVICES=$spec $*"
  CUDA_VISIBLE_DEVICES="$spec" $PY scripts/run_spatial_temporal_budget_30b.py "$@"
}

TWO="${FREE[0]},${FREE[1]}"
FOUR="$TWO"
if [ "${#FREE[@]}" -ge 4 ]; then
  FOUR="${FREE[0]},${FREE[1]},${FREE[2]},${FREE[3]}"
fi

SPEC="$TWO"
if ! run_with "$TWO" preflight; then
  echo "2-GPU BF16 load failed; retrying on $FOUR"
  if ! run_with "$FOUR" preflight; then
    echo "BF16 did not load on $FOUR" >&2
    exit 3
  fi
  SPEC="$FOUR"
fi
echo "using_gpus=$SPEC"

DRY=A_S0_fps0p5,A_S1_fps0p5,A_S2_fps0p5,C_S1_fps0p5,D_S1_fps0p5,D_S0_fps4
run_with "$SPEC" infer --ids "$DRY" --retry-failed
echo "==== full infer $(date -Is) ===="
run_with "$SPEC" infer --retry-failed
echo "==== aggregate $(date -Is) ===="
$PY scripts/run_spatial_temporal_budget_30b.py aggregate
echo "LAUNCH_DONE $(date -Is)"
