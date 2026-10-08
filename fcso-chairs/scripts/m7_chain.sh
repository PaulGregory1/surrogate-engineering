#!/bin/bash
# Milestone 7 run chain: required methods (gd, fcso_x0, gd_reg), then optional (fcso, fcso_noguide, fmg),
# each on all 10 opt chairs, one chair at a time (scripts/run_opt.py skips finished chairs and resumes the
# others from their checkpoints). Before each method, finished chairs that predate the saved evaluation
# meshes (meshes.npz) are re-evaluated so the report never decodes while a run is timing.
# Launch: nohup caffeinate -i scripts/m7_chain.sh > logs/m7_chain.stdout 2>&1 &
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=./.venv/bin/python
METHODS="${METHODS:-gd fcso_x0 gd_reg fcso fcso_noguide fmg}"
note() { echo "$(date '+%F %T') $*" >> logs/m7_chain.log; }
note "chain start: $METHODS"
for m in $METHODS; do
  miss=()
  for d in runs/$m/*/; do
    [ -f "$d/eval.json" ] && [ ! -f "$d/meshes.npz" ] && miss+=("$(basename "$d")")
  done
  if [ ${#miss[@]} -gt 0 ]; then
    note "$m: re-evaluating ${#miss[@]} chairs for meshes.npz"
    $PY scripts/run_opt.py --config configs/$m.yaml --ids "${miss[@]}" --re-eval >> logs/m7_$m.stdout 2>&1
  fi
  note "$m: start"
  $PY scripts/run_opt.py --config configs/$m.yaml >> logs/m7_$m.stdout 2>&1
  note "$m: exit $?"
  [ "$m" = "gd_reg" ] && note "REQUIRED RUNS DONE"
done
note "chain end"
