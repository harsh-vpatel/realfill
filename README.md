# RealFill

[RealFill](https://arxiv.org/abs/2309.16668) is a method to personalize text2image inpainting models like stable diffusion inpainting given just a few (1~5) images of a scene.
The `train_realfill.py` script shows how to implement the training procedure for stable diffusion inpainting. Since the diffusion model from stabilityai is deprecated, we will use sd2-community/stable-diffusion-2-inpainting instead.

The first part is modified from the forked github repo [by](https://github.com/thuanz123/realfill)


## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

cd to the realfill folder and run
```bash
cd realfill
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 

### Toy example

Now let's fill the real. For this example, we will use some images of the flower girl example from the paper.

We already provide some images for testing in data folder

You only have to launch the training using:

```bash
export MODEL_NAME="sd2-community/stable-diffusion-2-inpainting"
export TRAIN_DIR="data/flowerwoman"
export OUTPUT_DIR="flowerwoman-model"

accelerate launch train_realfill.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \
  --prompt_dropout_prob=0.1 \
  --mask_dropout_prob=0.1
```

### Training on a low-memory GPU:

It is possible to run realfill on a low-memory GPU by using the following optimizations:
- [gradient checkpointing and the 8-bit optimizer](#training-with-gradient-checkpointing-and-8-bit-optimizers)
- [xformers](#training-with-xformers)
- [setting grads to none](#set-grads-to-none)

```bash
export MODEL_NAME="sd2-community/stable-diffusion-2-inpainting"
export TRAIN_DIR="data/flowerwoman"
export OUTPUT_DIR="flowerwoman-model"

accelerate launch train_realfill.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \
  --prompt_dropout_prob=0.1 \
  --mask_dropout_prob=0.1
```

### Training with gradient checkpointing and 8-bit optimizers:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train realfill on a 16GB GPU.

To install `bitsandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script.

### Set grads to none

To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html


# RealFill: Experiment for Weighted Mask Loss

Extra argument added inside train_realfill_newloss.py, use train_realfill_newloss.py for training instead and add a multiplier (defaulted to 5.0).
The multiplier will increase the weight of the loss inside synthetic mask by the multiplier value.
```bash
-mask_loss_multiplier 5
```

Example:
```bash
export MODEL_NAME=""sd2-community/stable-diffusion-2-inpainting" "
export TRAIN_DIR="data/flowerwoman"
export OUTPUT_DIR="flowerwoman-model"
!accelerate launch train_realfill_newloss.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --mixed_precision=fp16 \
  --allow_tf32 \
  --gradient_checkpointing \
  --set_grads_to_none \
  --unet_learning_rate=2e-4 \
  --text_encoder_learning_rate=4e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --validation_steps=500 \
  --num_validation_images=2 \
  --checkpointing_steps=500 \
  --lora_rank=8 \
  --lora_dropout=0.1 \
  --lora_alpha=16 \
  --prompt_dropout_prob=0.1 \
  --mask_dropout_prob=0.1 \
  --mask_loss_multiplier 5
```
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

## Acknowledge
This repo is built upon the code of DreamBooth from diffusers and we thank the developers for their great works and efforts to release source code. Furthermore, a special "thank you" to RealFill's authors for publishing such an amazing work.
