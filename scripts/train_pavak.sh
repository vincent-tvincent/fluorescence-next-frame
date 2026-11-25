#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 DATA_ROOT NEW_RUN_DIRECTORY" >&2
    exit 2
fi

data=$(cd -- "$1" && pwd -P)
run=$2
if [[ $run != /* ]]; then run="$PWD/$run"; fi
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
python=${PYTHON:-python3}
train=()
for pos in {0..8}; do
    movie="$data/Pos$pos/SPIMA"
    if [[ ! -d $movie ]]; then
        echo "Missing movie directory: $movie" >&2
        exit 2
    fi
    if (( pos < 7 )); then train+=("$movie"); fi
done

mkdir -p -- "$(dirname -- "$run")"
mkdir -- "$run"
cd -- "$repo"
exec > >(tee "$run/console.log") 2>&1
echo "Started $(date -u +%FT%TZ), commit $(git rev-parse --short HEAD)"
echo "Data: $data"
echo "Output: $run"
"$python" -u -m fluoroframe train --train "${train[@]}" --val "$data/Pos7/SPIMA" \
    --run "$run/training" --device cuda --epochs 200 --steps-per-epoch 64 \
    --batch-size auto --vram-reserve-gib 1 --histories 1 2 4 8 --patch 12 96 96 --log-every 8

echo "Scoring the Pos7-selected checkpoint on the held-out Pos8 movie."
"$python" -u -m fluoroframe evaluate --device cuda --checkpoint "$run/training/best.pt" \
    --movies "$data/Pos8/SPIMA" --histories 1 4 8 --samples 100 | tee "$run/pos8.json.tmp"
mv -- "$run/pos8.json.tmp" "$run/pos8.json"
echo "Finished $(date -u +%FT%TZ). Scores: $run/pos8.json"
