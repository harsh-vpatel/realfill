# RealFill Inference Improvements

This repository provides **four inference-time improvements** for a trained **RealFill** model.

**Training is unchanged**. All methods operate only during inference and can be applied to existing RealFill checkpoints.

## Included Programs

1. **Pixel-wise Median Consensus (PMC)** — `infer_median.py`
2. **Concentric Boundary-to-Interior Filling (CBI)** — `infer_ring.py`
3. **Residual Refilling** — `infer_residual_refilling.py`
4. **Candidate Reranking** — `infer_rerank.py`

---

## Assumptions

- You already have a trained/exported RealFill model locally.
- Masks follow this convention:
  - **white / 255** = region to fill
  - **black / 0** = known region to preserve
- Default inference resolution is **512 × 512**.
- The model is loaded locally with `local_files_only=True`.

---

## Basic Dependencies

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install numpy opencv-python pillow tqdm
pip install kornia
```

Optional, only needed for some reranking features:

```bash
pip install pyiqa
```

---

## Expected Input Structure

```text
realfill_dataset/
└── RealBench/
    └── 21/
        ├── ref/         
        └── target/
            ├── target.png
            └── mask.png
```

---

# 1. Pixel-wise Median Consensus

**Script:** `infer_median.py`

## What It Does

Generates multiple stochastic inpainting outputs and computes a **pixel-wise median** inside the masked region to reduce seed-dependent artifacts.

## Run

```bash
python infer_median.py \
  --model_dir PATH_TO_MODEL \
  --train_data_dir PATH_TO_SCENE \
  --output_dir OUTPUT_DIR
```

## Common Options

- `--num_images` default: `16`
- `--num_inference_steps` default: `50`
- `--guidance_scale` default: `1.0`
- `--seed` default: `42`
- `--reference_dir` optional, for LoFTR scoring
- `--scores_json` optional
- `--save_variance_map` optional

## Example

```bash
python infer_median.py \
  --model_dir "bench21-model" \
  --train_data_dir "realfill_dataset/RealBench/21" \
  --output_dir "bench21-median-16" \
  --num_images 16 \
  --reference_dir "realfill_dataset/RealBench/21/ref" \
  --scores_json "bench21-median-16/scores.json" \
  --save_variance_map
```

## Output

- `00.png` — final median result
- `uncertainty_map.png` — optional
- `scores.json` — optional

---

# 2. Concentric Boundary-to-Interior Filling

**Script:** `infer_ring.py`

## What It Does

Fills the masked region progressively from the boundary inward using multiple inpainting passes.

## Run

```bash
python infer_ring.py \
  --model_dir PATH_TO_MODEL \
  --validation_image PATH_TO_TARGET \
  --validation_mask PATH_TO_MASK \
  --output_dir OUTPUT_DIR
```

## Common Options

- `--num_images` default: `16`
- `--n_rings` auto-selected if omitted
- `--ring_kernel_size` default: `24`
- `--reference_dir` optional
- `--scores_json` optional
- `--top_k` optional
- `--debug` optional

## Example

```bash
python infer_ring.py \
  --model_dir "bench0-model" \
  --validation_image "realfill_dataset/RealBench/0/target/target.png" \
  --validation_mask "realfill_dataset/RealBench/0/target/mask.png" \
  --reference_dir "realfill_dataset/RealBench/0/ref" \
  --output_dir "bench0-ring-16ranked" \
  --num_images 16 \
  --n_rings 4 \
  --scores_json "bench0-ring-16ranked/scores.json"
```

## Output

- `00.png`, `01.png`, ...
- `scores.json` — optional
- `debug/` — optional

---

# 3. Residual Refilling

**Script:** `infer_residual_refilling.py`

## What It Does

Refines only the most uncertain subregions of an already-ranked output using a second inpainting pass.

## Required Input

This script expects ranked candidate images named:

```text
00.png
01.png
02.png
...
```

and a corresponding `scores.json`.

## Run

```bash
python infer_residual_refilling.py \
  --model_dir PATH_TO_MODEL \
  --validation_mask PATH_TO_MASK \
  --ranked_images_dir PATH_TO_RANKED_IMAGES \
  --scores_json PATH_TO_SCORES_JSON \
  --output_dir OUTPUT_DIR
```

## Common Options

- `--residual_steps` default: `25`
- `--uncertainty_method` choices: `mad`, `std`
- `--uncertainty_quantile` default: `0.90`
- `--reference_dir` optional

## Example

```bash
python infer_residual_refilling.py \
  --model_dir "bench4-model" \
  --validation_mask "realfill_dataset/RealBench/4/target/mask.png" \
  --ranked_images_dir "bench4-32ranked_top16" \
  --scores_json "bench4-32ranked_top16/scores.json" \
  --output_dir "bench4-ramr2" \
  --uncertainty_quantile 0.95
```

## Output

- `00_base.png`
- `01_uncertainty_map.png`
- `02_residual_mask.png`
- `04_final.png`
- `summary.json`

---

# 4. Candidate Reranking

**Script:** `infer_rerank.py`

## What It Does

Reranks existing candidate images using a non-learned structural shortlist, with optional semantic and perceptual cues.

## Run

```bash
python infer_rerank.py \
  --input_dir PATH_TO_CANDIDATES \
  --validation_mask PATH_TO_MASK \
  --output_dir OUTPUT_DIR
```

## Common Options

- `--input_scores_json` optional
- `--reference_dir` optional
- `--top_k` default: `16`
- `--use_dino_ref` optional
- `--use_topiq_nr` optional
- `--use_maniqa` optional
- `--use_clipiqa` optional
- `--shortlist_size` default: `8`

## Example

```bash
python infer_rerank.py \
  --input_dir "bench31-32ranked_top16" \
  --input_scores_json "bench31-32ranked_top16/scores.json" \
  --validation_mask "realfill_dataset/RealBench/31/target/mask.png" \
  --reference_dir "realfill_dataset/RealBench/31/ref" \
  --output_dir "bench31-reranked" \
  --use_dino_ref \
  --shortlist_size 8
```

## Output

- `00.png`, `01.png`, ...
- `scores.json`
- `rerank_features_full.json`
