# Closed-loop evaluation with AlpaSim

The paper's closed-loop numbers come from [AlpaSim](https://github.com/NVlabs/alpasim), the public
sync of 2026-07-12 (AlpaSim base image 0.111.0). Setup:
- `pair_a15` topology, 2 rollouts per scene, driver threads 8, camera images not logged.
- Scenes are launched by NuRec uuid from the `public_2601` suite (release 26.01).

## Install

```bash
git clone https://github.com/NVlabs/alpasim && cd alpasim   # follow its README to build the images
python /path/to/MoOP/alpasim/install.py --alpasim .         # add --handover-fix, see below
```

`install.py` does three things (by hand, if your AlpaSim version differs):
1. Copies `alpamayo1_5_slim.py` into `src/driver/src/alpasim_driver/models/`.
2. In `alpamayo1_5_model.py`, imports `is_slim_checkpoint, load_slim` from it and loads with
   `load_slim(checkpoint_path, device=device, attn_implementation=...)` when the checkpoint directory
   holds `slim_meta.json`.
3. Adds the `pair_a15` topology.

The driver runs offline (`HF_HUB_OFFLINE=1`), so the Hugging Face cache mounted into the containers
must already hold four repositories. Running `scripts/eval_openloop.py` once fills the cache with
them.
- `nvidia/Alpamayo-1.5-10B` at the pinned revision
- the processor and config files of `nvidia/Cosmos-Reason2-8B`
- the processor and config files of `Qwen/Qwen3-VL-8B-Instruct`
- the processor and config files of `Qwen/Qwen3-VL-2B-Instruct`

## Hand-over braking fix

When the simulator hands control from the recorded drive to the model, the 2026-07-12 AlpaSim braked
the ego at about −8 m/s². `handover_fix.patch` backports the upstream fix from NVlabs/alpasim main
(2026-08-11).

Which harness produced which result:

| results | harness |
|---|---|
| paper tables (24% budget) | 2026-07-12 AlpaSim **without** the fix |
| 30% budget closed-loop runs | **with** the fix |

The fix changes absolute scores; for example, the dense model on Origin150 goes from 0.750 to 0.780. So
compare arms only within one harness version.

## Run and summarise

```bash
ALPASIM=/path/to/alpasim RUNS=runs/alpasim SCENE_CACHE=/path/to/nre-artifacts \
    bash alpasim/run_closed_loop.sh runs/moop_u40 data/scenes/hard100.csv moop_u40_hard100 "0 1 2 3"
python scripts/merge_alpasim_shards.py --runs-root runs/alpasim --out moop_u40_hard100 --expect-scenes 100 \
    --shards moop_u40_hard100_sh0 moop_u40_hard100_sh1 moop_u40_hard100_sh2 moop_u40_hard100_sh3
python scripts/summarize_closedloop.py --arms moop_u40=runs/alpasim/moop_u40_hard100 ... --ref moop_u40
```

`summarize_closedloop.py` reports two kinds of numbers.
- **Scene-score family.** Scene score, at-fault collision, off-road and progress, with scene-cluster
  bootstrap CIs. Rollouts that AlpaSim aborts count as 0 with no event, as in the paper.
- **AlpaSim score.** Kilometres driven per event (Alpamayo-R1 Sec. 6.1), both at-fault and all events.

Things to know:
- **Silent retries.** AlpaSim retries a failed rollout up to twice and logs `Rollout FAILED` for each
  attempt.
- **Hard100 start-up aborts.** Two scenes fail AlpaSim's route sanity check at start-up in every arm:
  `clipgt-4bad2f63…` and `clipgt-adb899bd…`.
- **Port collisions.** Shards started within seconds of each other can collide on host ports. The
  launcher staggers them.
