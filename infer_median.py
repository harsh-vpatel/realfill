
import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from tqdm import tqdm

import kornia as K
from torchvision import transforms
from kornia.feature import LoFTR


VALID_EXTS = {".png", ".jpg", ".jpeg"}

match_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def pil_to_loftr_gray(pil_img, device):
    t = match_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    return K.color.rgb_to_grayscale(t)


def masked_candidate_for_matching(gen_img, binary_mask_img):
    black = Image.new("RGB", gen_img.size, (0, 0, 0))
    return Image.composite(gen_img, black, binary_mask_img)


def correspondence_score(gen_img, ref_paths, binary_mask_img, threshold, loftr, device):
    image0 = pil_to_loftr_gray(masked_candidate_for_matching(gen_img, binary_mask_img), device)
    total_matches = 0

    with torch.no_grad():
        for ref_path in ref_paths:
            ref_img = Image.open(ref_path).convert("RGB")
            image1 = pil_to_loftr_gray(ref_img, device)

            out = loftr({"image0": image0, "image1": image1})

            if "confidence" in out:
                total_matches += int((out["confidence"] > threshold).sum().item())
            else:
                total_matches += int(out["keypoints0"].shape[0])

    return total_matches


def compute_pmc(
    stack: np.ndarray,
    mask_np: np.ndarray,
    original_np: np.ndarray,
    sharpen_alpha: float = 0.4,
    sharpen_radius: int = 3,
) -> np.ndarray:
    """
    Compute Pixel-wise Median Consensus over a stack of generated images.

    Args:
        stack:          (N, H, W, 3) float32 array of all N generated images
                        in the range [0, 255].
        mask_np:        (H, W) uint8 mask — 255 = fill region, 0 = keep region.
        original_np:    (H, W, 3) float32 original target image.
        sharpen_alpha:  strength of unsharp masking applied inside the mask.
                        0.0 = no sharpening.
        sharpen_radius: Gaussian kernel radius for unsharp mask (pixels).

    Returns:
        (H, W, 3) uint8 consensus image.
    """
    assert stack.ndim == 4 and stack.shape[-1] == 3, \
        f"stack must be (N, H, W, 3), got {stack.shape}"
    assert stack.shape[1:3] == mask_np.shape[:2], \
        "stack and mask spatial dimensions must match"

    mask_bool = mask_np > 127

    # Step 1: pixel-wise channel-wise median
    consensus = np.median(stack, axis=0)  # (H, W, 3), float64

    # Step 2: optional sharpening inside the mask
    if sharpen_alpha > 0.0:
        ksize = 2 * sharpen_radius + 1
        blurred = cv2.GaussianBlur(
            consensus.astype(np.float32),
            (ksize, ksize),
            sigmaX=sharpen_radius / 2.0,
        )
        sharpened = consensus + sharpen_alpha * (consensus - blurred.astype(np.float64))
        sharpened = np.clip(sharpened, 0.0, 255.0)
        consensus = np.where(mask_bool[:, :, np.newaxis], sharpened, consensus)

    # Step 3: preserve original pixels outside mask
    result = np.where(
        mask_bool[:, :, np.newaxis],
        consensus,
        original_np.astype(np.float64),
    )

    return result.clip(0, 255).astype(np.uint8)


def compute_variance_map(stack: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel standard deviation across N samples inside the mask.
    """
    mask_bool = mask_np > 127
    std_map = np.std(stack, axis=0)      # (H, W, 3)
    mean_std = std_map.mean(axis=-1)     # (H, W)
    mean_std[~mask_bool] = 0.0
    return mean_std.astype(np.float32)


def save_variance_map(var_map: np.ndarray, save_path: str) -> None:
    """
    Save uncertainty map as heatmap PNG.
    """
    if var_map.max() < 1e-6:
        Image.fromarray(np.zeros_like(var_map, dtype=np.uint8)).save(save_path)
        return
    normalised = ((var_map / var_map.max()) * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    Image.fromarray(heatmap_rgb).save(save_path)



def generate_stack(
    pipeline: StableDiffusionInpaintPipeline,
    target_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    base_seed: int,
    device: str,
) -> np.ndarray:
    """
    Generate `num_images` independent samples using the pipeline.

    Returns:
        (num_images, H, W, 3) float32 array in [0, 255].
    """
    images = []

    for i in tqdm(range(num_images), desc="Generating samples", leave=False):
        gen = torch.Generator(device=device).manual_seed(base_seed + i)
        result = pipeline(
            prompt=prompt,
            image=target_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        result_np = np.array(result, dtype=np.float32)
        images.append(result_np)

    return np.stack(images, axis=0)



def parse_args():
    p = argparse.ArgumentParser(
        description="PMC-only inference for RealFill."
    )
    p.add_argument("--model_dir", type=str, required=True,
                   help="Fine-tuned RealFill model directory.")
    p.add_argument("--train_data_dir", type=str, required=True,
                   help="Train data dir used during training (contains target/target.png and target/mask.png).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory.")
    p.add_argument("--num_images", type=int, default=16,
                   help="Number of stochastic samples to generate.")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM steps per sample.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale. RealFill default is 1.")
    p.add_argument("--prompt", type=str, default="a photo of sks")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed. Sample i uses seed+i.")
    p.add_argument("--resolution", type=int, default=512,
                   help="Inference resolution. Default 512 to match training.")
    p.add_argument("--sharpen_alpha", type=float, default=0.4,
                   help="Unsharp mask strength inside the fill region. 0 = disabled.")
    p.add_argument("--sharpen_radius", type=int, default=3,
                   help="Gaussian kernel radius for unsharp masking.")
    p.add_argument("--save_variance_map", action="store_true", default=False,
                   help="Save a heatmap of per-pixel sample uncertainty.")
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])

    # Added back from original-style workflow
    p.add_argument("--reference_dir", type=str, default=None,
                   help="Optional reference image directory for LoFTR-based ranking of raw candidates.")
    p.add_argument("--match_threshold", type=float, default=0.5,
                   help="Confidence threshold for counting LoFTR correspondences.")
    p.add_argument("--scores_json", type=str, default=None,
                   help="Optional path to save ranked raw-candidate scores as JSON.")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map[args.mixed_precision]

    # Load pipeline
    print(f"Loading model: {args.model_dir}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=weight_dtype,
        local_files_only=True,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Load inputs
    target_dir = Path(args.train_data_dir) / "target"
    target_image = Image.open(target_dir / "target.png").convert("RGB")
    mask_image = Image.open(target_dir / "mask.png").convert("L")

    # FIX: resize to inference/training resolution before generation
    inference_size = (args.resolution, args.resolution)
    target_image = target_image.resize(inference_size, Image.LANCZOS)
    mask_image = mask_image.resize(inference_size, Image.NEAREST)

    mask_np = np.array(mask_image, dtype=np.uint8)
    original_np = np.array(target_image, dtype=np.float32)
    binary_mask_image = mask_image.point(lambda p: 255 if p > 127 else 0)

    mask_coverage = float((mask_np > 127).mean())
    print(f"Mask coverage: {mask_coverage:.1%}")
    print(f"Generating {args.num_images} independent samples...")

    # Optional LoFTR ranking setup
    loftr = None
    ref_paths = []
    if args.reference_dir is not None:
        loftr = LoFTR(pretrained="outdoor").to(device).eval()
        ref_paths = sorted([
            str(p) for p in Path(args.reference_dir).iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ])
        print(f"Loaded {len(ref_paths)} reference images for LoFTR ranking")

    # Generate all N samples
    stack = generate_stack(
        pipeline=pipeline,
        target_image=target_image,
        mask_image=mask_image,
        prompt=args.prompt,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.seed,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Optional raw candidate scoring/ranking
    if args.scores_json is not None:
        candidates = []
        for i in range(stack.shape[0]):
            raw_img = Image.fromarray(stack[i].clip(0, 255).astype(np.uint8))

            # Match the original workflow: score composited image, not raw whole-frame output
            composited = Image.composite(raw_img, target_image, binary_mask_image)

            score = None
            if loftr is not None and len(ref_paths) > 0:
                score = correspondence_score(
                    composited,
                    ref_paths,
                    binary_mask_image,
                    args.match_threshold,
                    loftr,
                    device,
                )

            candidates.append({
                "candidate_idx": i,
                "score": score,
            })

        if loftr is not None and len(ref_paths) > 0:
            candidates = sorted(
                candidates,
                key=lambda x: x["score"] if x["score"] is not None else -1,
                reverse=True,
            )

        scores_parent = Path(args.scores_json).parent
        if str(scores_parent) != "":
            scores_parent.mkdir(parents=True, exist_ok=True)

        with open(args.scores_json, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "rank": rank,
                        "candidate_idx": item["candidate_idx"],
                        "score": item["score"],
                    }
                    for rank, item in enumerate(candidates)
                ],
                f,
                indent=2,
            )

        print(f"Raw candidate scores saved to: {args.scores_json}")

    # Compute PMC
    print("Computing pixel-wise median consensus...")
    consensus_img = compute_pmc(
        stack=stack,
        mask_np=mask_np,
        original_np=original_np,
        sharpen_alpha=args.sharpen_alpha,
        sharpen_radius=args.sharpen_radius,
    )

    consensus_pil = Image.fromarray(consensus_img)
    consensus_pil.save(os.path.join(args.output_dir, "00.png"))
    print(f"PMC consensus image saved to: {args.output_dir}/00.png")

    # Optional uncertainty map
    if args.save_variance_map:
        var_map = compute_variance_map(stack, mask_np)
        var_path = os.path.join(args.output_dir, "uncertainty_map.png")
        save_variance_map(var_map, var_path)

        mean_uncertainty = float(var_map[mask_np > 127].mean()) if (mask_np > 127).any() else 0.0
        print(f"Mean per-pixel uncertainty inside mask: {mean_uncertainty:.2f} / 255")
        print(f"Uncertainty map saved to: {var_path}")

    print("\n── Configuration ──")
    print(f"  num_images        = {args.num_images}")
    print(f"  num_steps/sample  = {args.num_inference_steps}")
    print(f"  sharpen_alpha     = {args.sharpen_alpha}")
    print(f"  mask coverage     = {mask_coverage:.1%}")


if __name__ == "__main__":
    main()


"""
python infer_median.py `
  --model_dir="bench21-model" `
  --train_data_dir="realfill_dataset/RealBench/21" `
  --output_dir="bench21-pmc-16" `
  --num_images=16 `
  --num_inference_steps=50 `
  --guidance_scale=1.0 `
  --resolution=512 `
  --seed=42 `
  --sharpen_alpha=0.4 `
  --sharpen_radius=3 `
  --reference_dir="realfill_dataset/RealBench/21/ref" `
  --scores_json="bench21-pmc-16/scores.json" `
  --save_variance_map
"""
