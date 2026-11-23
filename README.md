# MoOP: Exploring What to Preserve for Reasoning and Action in Structured Pruning of Driving VLAs

Seongchan Kim¹, Soowon Son¹, Sangoh Kim¹, Hyunah Ko¹, Eunbeen Hong¹, Hyunkoo Lee¹, Sanghyun Lee¹,
Wonjun Moon¹, Kihong Kim², Dongyeon Son², Woongchan Nam², Seungryong Kim¹

¹KAIST · ²LG Innotek

[Project page](https://cvlab-kaist.github.io/MoOP/) · Paper (coming soon)

MoOP (Mixture-of-Objective Pruning) is a structured pruning recipe for reasoning vision–language–action
models, applied to NVIDIA [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) (10B parameters). It
needs no fine-tuning after pruning.

- **VLM.** Query heads and MLP channels are scored with Taylor importance under two objectives:
  - the Chain-of-Causation reasoning text (cross-entropy, CE)
  - the action expert's flow-matching loss (FM), which reaches the VLM only through the KV cache

  Within each layer, a unit's score is the maximum of its two ranks, and the lowest-scoring units are removed.
- **Action expert.** Per-step FM importance is aggregated over the denoising steps. A fixed subset of
  MLP channels is kept.

| | params removed | Test500 minADE@6 | OOD-Val minADE@6 | LingoQA | Origin150 scene score | latency | peak memory |
|---|---|---|---|---|---|---|---|
| Alpamayo 1.5 (dense) | 0% | 0.842 m | 1.000 m | 73.2 | 0.750 | 1.00× | 22.5 GiB |
| MoOP (VLM) | 24.0% | 0.950 m | 1.119 m | 68.8 | 0.828 | 1.30× faster | 17.4 GiB |
| MoOP + expert | 39.4% | 0.951 m | 1.118 m | – | 0.793 | 1.37× faster | 14.3 GiB |

## Install

Requires Python 3.12, CUDA 12.8 and [uv](https://docs.astral.sh/uv/). A 48 GB GPU covers importance
measurement (peak about 40 GB) and evaluation.

```bash
uv sync                      # torch 2.8.0, transformers 4.57.1, alpamayo1_5 @ f42e594, numpy 2.2.6
huggingface-cli login        # gated: nvidia/Alpamayo-1.5-10B, nvidia/PhysicalAI-Autonomous-Vehicles
```

All results use base weights `nvidia/Alpamayo-1.5-10B` at revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`
(pinned in `moop/__init__.py`).

## Quick start: prune with the paper's importance scores

`recipes/importance/` holds the importance scores the paper's models were cut from. `importance.npz`
is the VLM CE/FM scores on `calib_100`; `expert_importance.npz` is the expert step-aggregated scores.

```bash
python scripts/build_slim.py --recipe moop_u40 \
    --importance recipes/importance/importance.npz --out runs/moop_u40
python scripts/verify_recipe.py runs/moop_u40/slim_meta.json      # MATCH = the paper's kept units
```

| recipe | VLM cut per layer | expert MLP cut | removed | paper row |
|---|---|---|---|---|
| `moop_u40` | 13/32 Q heads, 4898/12288 MLP channels | – | 24.0% | MoOP |
| `moop_e_u40` | same | 7740/8256 per layer (needs `--expert-importance`) | 39.4% | MoOP + expert |
| `fm_only_u40`, `ce_only_u40` | same, ranked by one objective | – | 24.0% | FM-only, CE-only |
| `moop_u40_l0keep` | layer 0 keeps all Q heads; the same budget goes to MLP channels | – | 24.0% | (variant, see below) |
| `moop_u50`, `moop_e_u50`, `fm_only_u50`, `ce_only_u50` | 16/32 heads, 6144/12288 channels | – / 7740 | 30.0% / 45.4% | 30% budget |

A pruned model is `slim_meta.json`, the kept indices per layer. `moop.slim.load_slim(dir)` rebuilds it
from the pinned base weights, bit for bit, so no pruned checkpoint is distributed. `--save-state`
additionally writes the 17 GB state_dict.

## Reproduce the importance scores

```bash
python scripts/build_cache.py --manifest calib_100 --cache calib
python scripts/compute_importance.py --manifest calib_100 --cache calib --out runs/importance
python scripts/compute_expert_importance.py --manifest calib_100 --cache calib --out runs/expert_importance
# optional: --shard i --n-shards k on k GPUs, then scripts/merge_importance.py
python scripts/build_slim.py --recipe moop_e_u40 --importance runs/importance/importance.npz \
    --expert-importance runs/expert_importance/expert_importance.npz --out runs/moop_e_u40
```

Paper setting: 100 clips, seed 42, CoC sampled at temperature 0.6 / top-p 0.98 up to 256 tokens,
FM loss over 10 denoising steps at t = (s + 0.5) / 10. Per-clip seeds come from the clip id, so
sharding does not change results.

Two notes on reproducing the masks exactly:
- **Ties at the last VLM layer.** The FM scores of the last VLM layer are exactly zero: its outputs
  never reach the KV cache the expert reads. So which of its MLP channels an FM-based recipe cuts
  depends on `np.argsort`'s order among equal values. That order changed across numpy releases, which
  is why numpy is pinned to 2.2.6.
- **Layer 0 of the plain recipe.** Several layer-0 Q heads of the plain recipe tie in rank at the cut.
  A re-measured importance can therefore swap them, and on Test500 that alone moves minADE by up to
  ~0.8 m. `moop_u40_l0keep` removes this sensitivity at the same budget; the shipped scores
  reproduce the paper's masks either way.

## Evaluate

**Open loop** (Test500, OOD-Val, OOD-Train; minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s):
```bash
python scripts/build_cache.py --manifest test_500 --cache test
python scripts/build_cache.py --manifest ood_val --cache ood
python scripts/eval_openloop.py --model runs/moop_u40 --manifest test_500 --cache test --out runs/openloop/moop_u40
python scripts/eval_openloop.py --model dense         --manifest test_500 --cache test --out runs/openloop/dense
python scripts/summarize_openloop.py --manifest test_500 \
    --arms dense=runs/openloop/dense moop_u40=runs/openloop/moop_u40 --ref moop_u40
```
The model writes its own CoC (seeded per clip). The prompt and CoC are re-prefilled teacher-forced,
and 8 trajectories are denoised; minADE@6 uses the first 6. Kernels are deterministic, so results are
bit-identical on one GPU model and move by about ±0.02 m across GPU generations. Compare arms on one
GPU type.

**LingoQA** (zero-shot, 500 validation questions, judged by Lingo-Judge). This needs the
[LingoQA](https://github.com/wayveai/LingoQA) data and repository, which are not redistributed (Wayve
non-commercial licence):
```bash
LINGOQA_DATA=/path/to/lingoqa/val python scripts/eval_lingoqa.py --model runs/moop_u40 --out runs/lingoqa/moop_u40
LINGOQA_DATA=... LINGOQA_BENCH=/path/to/LingoQA/benchmark python scripts/score_lingoqa.py \
    --arms dense=runs/lingoqa/dense moop_u40=runs/lingoqa/moop_u40 --ref moop_u40
```

**Closed loop** ([AlpaSim](https://github.com/NVlabs/alpasim), Origin150 and Hard100, 2 rollouts per
scene). See [`alpasim/README.md`](alpasim/README.md):
```bash
python alpasim/install.py --alpasim /path/to/alpasim
ALPASIM=/path/to/alpasim RUNS=runs/alpasim SCENE_CACHE=/path/to/nre-artifacts \
    bash alpasim/run_closed_loop.sh runs/moop_u40 data/scenes/origin150.csv moop_u40_origin150 "0 1 2 3"
python scripts/merge_alpasim_shards.py --runs-root runs/alpasim --out moop_u40_origin150 \
    --shards moop_u40_origin150_sh0 moop_u40_origin150_sh1 moop_u40_origin150_sh2 moop_u40_origin150_sh3
python scripts/summarize_closedloop.py --arms moop_u40=runs/alpasim/moop_u40_origin150 dense=... --ref moop_u40
```

## Data

Only ids and timestamps are included: [`data/README.md`](data/README.md) describes every list and how it
was drawn. The calibration set is `calib_100`; the open-loop sets are Test500, OOD-Val and OOD-Train;
the closed-loop sets are Origin150 and Hard100.

## Licence

Code: Apache-2.0 (see `NOTICE`). The base model weights (OpenMDW-1.1), the PhysicalAI-AV dataset, the
AlpaSim scenes and LingoQA are not distributed here and keep their own licences.

This work was made using the Wayve LingoQA dataset and benchmark, provided by Wayve Technologies Ltd under licence
terms available at https://github.com/wayveai/LingoQA.

## Citation

```bibtex
@article{moop2026,
  title   = {Exploring What to Preserve for Reasoning and Action in Structured Pruning of Driving VLAs},
  author  = {Kim, Seongchan and Son, Soowon and Kim, Sangoh and Ko, Hyunah and Hong, Eunbeen and
             Lee, Hyunkoo and Lee, Sanghyun and Moon, Wonjun and Kim, Kihong and Son, Dongyeon and
             Nam, Woongchan and Kim, Seungryong},
  year    = {2026}
}
```
