# Sample lists

Every number in the paper comes from the fixed lists in this directory. Clip and scene ids refer to
NVIDIA's [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset and the [AlpaSim](https://github.com/NVlabs/alpasim) NuRec scene release 26.01 (gated; request
access on the hub). No data is redistributed here, only ids and timestamps.

| list | file | size | used for |
|---|---|---|---|
| Calibration | `manifests/calib_100.parquet` | 100 clips | Taylor importance (all recipes) |
| Test500 | `manifests/test_500.parquet` | 500 clips | open loop, in distribution; CoC NLL |
| OOD-Val | `manifests/ood_val.parquet` | 262 clips | open loop, out of distribution |
| OOD-Train | `manifests/ood_train.parquet` | 1,271 clips | open loop, out of distribution (larger sample) |
| Origin150 | `scenes/origin150.csv` | 150 scenes | closed loop (AlpaSim) |
| Hard100 | `scenes/hard100.csv` | 100 scenes | closed loop (AlpaSim) |

Open-loop manifests have `clip_id`, `t0_us` (start of the prediction window, microseconds into the
clip) and `chunk` (the dataset chunk holding the clip). `scripts/build_cache.py --manifest <name>`
turns a manifest into the sample cache the code reads.

## Calibration: `calib_100`
- 100 clips from the dataset's official **train** split, drawn with seed 42 by greedy distribution
  matching on six collection-metadata attributes (country, platform, radar configuration, month,
  hour of day, season; the extra columns in the manifest). The draw minimises a weighted L1 distance
  to the clip-level distribution of the whole train split (0.031, against 0.178 for a random draw of the
  same size).
- The sampler saw only collection metadata, never the driving.
- Window start t0 = 5.1 s for every clip.
- Disjoint from every evaluation list.

## Test500: `test_500`
- 500 clips from the official **test** split, same procedure and seed (weighted L1 0.006 against
  0.084 for a random draw).
- t0 = 5.1 s.

## OOD-Val and OOD-Train: `ood_val`, `ood_train`
- Taken from the dataset's human-verified out-of-distribution reasoning labels
  (`reasoning/ood_reasoning.parquet`).
- A clip enters when it has an event carrying both a start timestamp and curated Chain-of-Causation
  text. The sample starts at that event (`t0_us` = event start). The lists split the clips by the
  dataset's own `val` / `train` split, and `cluster` is the dataset's scenario cluster.
- `scripts/make_ood_manifest.py` produces the candidates: 289 val and 1,442 train clips.
- Candidates whose window the dataset loader cannot build were dropped. They are listed with the
  reason in `manifests/ood_excluded.csv`:
  - 195: the event starts too early in the clip for the 1.6 s history window
  - 3: the 6.4 s future window runs past the end of the clip
- The two lists do not overlap each other or `calib_100`.

## Origin150: `scenes/origin150.csv`
- The first 150 scenes, by `scene_id`, of AlpaSim's `public_2601` suite (913 scenes).
- Each row carries the scene's NuRec `uuid`. Always launch by uuid: 159 of the suite's `scene_id`s
  also exist in the newer 26.04 release, and resolving a bare id silently picks the newer render.

## Hard100: `scenes/hard100.csv`
- 100 scenes of `public_2601` outside Origin150, chosen as the hardest by a score computed from the
  recorded ego track alone: `z(mean speed) + z(total yaw change)`.
- Scenes whose recorded ego travels less than 5 m are excluded. AlpaSim's progress score is fixed to
  1 for those, so they carry no information.
- On an independent 913-scene run of the unpruned model, the score correlates with the closed-loop
  scene score at Spearman ρ = −0.29.
- In every arm, AlpaSim aborts two of these scenes at start-up (route sanity check: the recorded route
  folds back on itself): `clipgt-4bad2f63…` and `clipgt-adb899bd…`. The paper scores aborted rollouts as
  0 with no event.
