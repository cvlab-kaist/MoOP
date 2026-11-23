#!/bin/bash
# Closed-loop evaluation of one model on a scene list with AlpaSim, one shard per GPU.
#
#   ALPASIM=/path/to/alpasim RUNS=runs/alpasim SCENE_CACHE=/path/to/nre-artifacts \
#   bash alpasim/run_closed_loop.sh <model_dir|dense> <scenes.csv> <run_name> "<gpu ids>"
#
#   e.g. bash alpasim/run_closed_loop.sh runs/moop_u40 data/scenes/origin150.csv moop_u40_origin150 "0 1 2 3"
#
# <model_dir> holds slim_meta.json (scripts/build_slim.py); it is mounted read-only into the driver
# container. 2 rollouts per scene, as in the paper. The base weights and the processors the driver loads
# offline (nvidia/Alpamayo-1.5-10B, nvidia/Cosmos-Reason2-8B, Qwen/Qwen3-VL-8B-Instruct,
# Qwen/Qwen3-VL-2B-Instruct) must already be in the HF cache mounted into the containers.
# After all shards finish: scripts/merge_alpasim_shards.py, then scripts/summarize_closedloop.py.
set -u
MODEL=${1:?model dir or dense}
SCENES=${2:?scenes csv}
NAME=${3:?run name}
read -r -a GPUS <<<"${4:?gpu list}"
ALPASIM=${ALPASIM:?set ALPASIM to the AlpaSim checkout}
RUNS=${RUNS:?set RUNS to the output root}
SCENE_CACHE=${SCENE_CACHE:?set SCENE_CACHE to the NuRec artifact cache}
N_ROLLOUTS=${N_ROLLOUTS-2}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$RUNS"
RUNS=$(cd "$RUNS" && pwd)

SUITES="$RUNS/suites_${NAME}.csv"
mapfile -t SHARDS < <("$REPO/.venv/bin/python" "$REPO/scripts/make_shard_suites.py" --scenes "$SCENES" \
  --k "${#GPUS[@]}" --name "$NAME" --sim-suites "$ALPASIM/data/scenes/sim_suites.csv" --out "$SUITES")
[ "${#SHARDS[@]}" -eq 0 ] && { echo "suite generation failed" >&2; exit 1; }

extra=()
if [ "$MODEL" != "dense" ]; then
  MODEL=$(cd "$MODEL" && pwd)
  extra+=("defines.drivers=$(dirname "$MODEL")" "driver.model.checkpoint_path=/mnt/drivers/$(basename "$MODEL")")
fi

pids=()
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  read -r suite n <<<"${SHARDS[$i]}"
  [ "${n:-0}" -eq 0 ] && continue
  (
    cd "$ALPASIM" &&
    uv run alpasim_wizard \
      deploy=local topology=pair_a15 driver=alpamayo1_5 \
      wizard.log_dir="$RUNS/${NAME}_sh${i}" wizard.run_name="${NAME}_sh${i}" \
      "services.renderer.gpus=[$gpu]" "services.physics.gpus=[$gpu]" \
      "services.trafficsim.gpus=[$gpu]" "services.driver.gpus=[$gpu]" \
      "services.driver.environments=['HF_HUB_OFFLINE=1','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8']" \
      "services.runtime.environments=['ALPASIM_ASL_SKIP_IMAGES=1']" \
      scenes.scene_ids=null scenes.test_suite_id="$suite" \
      "scenes.suites_csv=[$ALPASIM/data/scenes/sim_suites.csv,$SUITES]" \
      scenes.scene_cache="$SCENE_CACHE" \
      runtime.simulation_config.n_rollouts="$N_ROLLOUTS" \
      eval.allow_aggregation_with_failed_rollouts=true eval.video.render_video=false \
      "${extra[@]}"
  ) > "$RUNS/${NAME}_sh${i}.launch.log" 2>&1 &
  pids+=($!)
  echo "shard $i -> GPU $gpu, $n scenes, log $RUNS/${NAME}_sh${i}.launch.log"
  # the wizard picks free host ports when it writes the compose file; stagger so shards do not collide
  sleep 60
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "$NAME finished (fail=$fail); merge with:"
echo "  python scripts/merge_alpasim_shards.py --runs-root $RUNS --out $NAME --shards $(for i in "${!GPUS[@]}"; do printf '%s ' "${NAME}_sh$i"; done)"
exit $fail
