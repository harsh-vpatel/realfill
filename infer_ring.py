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


def build_erosion_rings(mask_np: np.ndarray, n_rings: int, kernel_size: int):
    """
    Decompose a binary mask into concentric rings via morphological erosion.

    Args:
        mask_np: H×W uint8, white(255)=unknown region to fill, black(0)=known
        n_rings: desired number of rings
        kernel_size: erosion kernel diameter in pixels

    Returns:
        List of H×W uint8 ring masks ordered outermost → innermost
    """
    binary = (mask_np > 127).astype(np.uint8)

    if binary.max() == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    rings = []
    prev = binary.copy()

    for _ in range(max(n_rings - 1, 0)):
        eroded = cv2.erode(prev, kernel, iterations=1)
        ring = (prev - eroded).clip(0, 1)
        if ring.max() > 0:
            rings.append((ring * 255).astype(np.uint8))
        prev = eroded
        if prev.max() == 0:
            break

    if prev.max() > 0:
        rings.append((prev * 255).astype(np.uint8))

    return rings


def visualise_rings(mask_np: np.ndarray, rings, save_path=None):
    colours = [
        (255, 100, 100),
        (255, 200, 100),
        (100, 255, 100),
        (100, 100, 255),
        (200, 100, 255),
        (100, 255, 255),
    ]
    vis = np.zeros((*mask_np.shape[:2], 3), dtype=np.uint8)
    for i, ring in enumerate(rings):
        colour = colours[i % len(colours)]
        for c, v in enumerate(colour):
            vis[:, :, c] = np.where(ring > 127, v, vis[:, :, c])

    if save_path:
        Image.fromarray(vis).save(save_path)
    return vis


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


def run_inpaint_pass(
    pipeline,
    image_pil,
    mask_pil,
    prompt,
    num_inference_steps,
    guidance_scale,
    generator,
):
    return pipeline(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]


def cbi_fill(
    pipeline,
    target_image,
    full_mask,
    prompt,
    n_rings,
    ring_kernel_size,
    num_inference_steps,
    guidance_scale,
    generator,
    debug_dir=None,
):
    """
    Practical approximation of boundary-to-center filling:

    At pass k:
      - mask the full remaining unknown region
      - run one inpainting pass
      - accept only the current outer ring
      - defer deeper rings to later passes

    This is the closest implementation compatible with the stock
    StableDiffusionInpaintPipeline interface.
    """
    mask_np = np.array(full_mask.convert("L"))
    target_np = np.array(target_image.convert("RGB")).astype(np.float32)

    rings = build_erosion_rings(mask_np, n_rings=n_rings, kernel_size=ring_kernel_size)

    if not rings:
        return target_image

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        visualise_rings(mask_np, rings, save_path=os.path.join(debug_dir, "rings.png"))

    canvas_np = target_np.copy()

    # known_np = pixels already trusted/fixed
    known_np = (mask_np <= 127)

    for ring_idx, ring_np in enumerate(rings):
        ring_bool = ring_np > 127
        if not ring_bool.any():
            continue

        # remaining unknown region = current ring + all deeper rings
        remaining_unknown = ~known_np
        inference_mask_np = (remaining_unknown.astype(np.uint8) * 255)

        canvas_pil = Image.fromarray(canvas_np.clip(0, 255).astype(np.uint8))
        inference_mask_pil = Image.fromarray(inference_mask_np)

        result_pil = run_inpaint_pass(
            pipeline,
            canvas_pil,
            inference_mask_pil,
            prompt,
            num_inference_steps,
            guidance_scale,
            generator,
        )
        result_np = np.array(result_pil).astype(np.float32)

        # commit only current ring
        for c in range(3):
            canvas_np[:, :, c] = np.where(ring_bool, result_np[:, :, c], canvas_np[:, :, c])

        known_np = known_np | ring_bool

        if debug_dir:
            Image.fromarray(canvas_np.clip(0, 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"ring_{ring_idx:02d}_result.png")
            )

    # preserve all originally-known pixels exactly
    final_np = target_np.copy()
    orig_known = (mask_np <= 127)
    for c in range(3):
        final_np[:, :, c] = np.where(orig_known, target_np[:, :, c], canvas_np[:, :, c])

    return Image.fromarray(final_np.clip(0, 255).astype(np.uint8))



def mask_coverage(mask_pil):
    arr = np.array(mask_pil.convert("L"))
    return float((arr > 127).sum()) / arr.size


def recommended_n_rings(coverage, kernel_size, image_size=512):
    import math
    radius_px = math.sqrt(coverage) * image_size / 2
    n = max(2, math.ceil(radius_px / kernel_size))
    return min(n, 8)



def parse_args():
    p = argparse.ArgumentParser(description="CBI-only inference aligned with improved RealFill training.")

    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to final exported improved RealFill model.")
    p.add_argument("--validation_image", type=str, required=True,
                   help="Path to target.png")
    p.add_argument("--validation_mask", type=str, required=True,
                   help="Path to mask.png")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save generated images")

    p.add_argument("--reference_dir", type=str, default=None,
                   help="Optional reference image directory for LoFTR-based ranking.")
    p.add_argument("--match_threshold", type=float, default=0.5,
                   help="Confidence threshold for counting LoFTR correspondences.")
    p.add_argument("--scores_json", type=str, default=None,
                   help="Optional path to save ranked correspondence scores as JSON.")
    p.add_argument("--top_k", type=int, default=None,
                   help="Number of top-ranked outputs to keep. If omitted, save all generated images.")

    p.add_argument("--n_rings", type=int, default=None,
                   help="Number of rings. If omitted, auto-selected from mask coverage.")
    p.add_argument("--ring_kernel_size", type=int, default=24,
                   help="Ring width controller in pixels.")
    p.add_argument("--num_images", type=int, default=16,
                   help="Number of generated candidate images.")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="DDIM steps per pass. Default 50 to match improved training validation.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale. RealFill uses 1.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default="a photo of sks")
    p.add_argument("--resolution", type=int, default=512,
                   help="Inference resolution. Default 512 to match training.")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.mixed_precision]

    print(f"Loading model from: {args.model_dir}")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=weight_dtype,
        local_files_only=True,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Load optional LoFTR matcher
    loftr = None
    ref_paths = []
    if args.reference_dir is not None:
        loftr = LoFTR(pretrained="outdoor").to(device).eval()
        ref_paths = sorted([
            str(p) for p in Path(args.reference_dir).iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ])
        print(f"Loaded {len(ref_paths)} reference images for LoFTR ranking")

    # align with improved training: raw image + raw mask, resized to 512 by default
    target_image = Image.open(args.validation_image).convert("RGB")
    mask_image = Image.open(args.validation_mask).convert("L")

    inference_size = (args.resolution, args.resolution)
    target_image = target_image.resize(inference_size, Image.LANCZOS)
    mask_image = mask_image.resize(inference_size, Image.NEAREST)

    binary_mask_image = mask_image.point(lambda p: 255 if p > 127 else 0)

    coverage = mask_coverage(mask_image)
    print(f"Mask coverage: {coverage:.1%} of image")

    if args.n_rings is None:
        n_rings = recommended_n_rings(coverage, args.ring_kernel_size, image_size=args.resolution)
        print(f"Auto-selected n_rings = {n_rings}")
    else:
        n_rings = args.n_rings
        print(f"Using n_rings = {n_rings}")

    mask_np = np.array(mask_image)
    rings = build_erosion_rings(mask_np, n_rings=n_rings, kernel_size=args.ring_kernel_size)
    print(f"Actual rings constructed: {len(rings)}")
    for i, r in enumerate(rings):
        px = (r > 127).sum()
        print(f"  Ring {i}: {px:,} pixels ({px / mask_np.size:.1%} of image)")

    os.makedirs(args.output_dir, exist_ok=True)

    candidates = []

    for i in tqdm(range(args.num_images), desc="Generating"):
        gen_i = torch.Generator(device=device).manual_seed(args.seed + i)

        debug_dir = None
        if args.debug:
            debug_dir = os.path.join(args.output_dir, "debug", f"img_{i:02d}")

        out = cbi_fill(
            pipeline=pipeline,
            target_image=target_image,
            full_mask=mask_image,
            prompt=args.prompt,
            n_rings=n_rings,
            ring_kernel_size=args.ring_kernel_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen_i,
            debug_dir=debug_dir,
        )

        score = None
        if loftr is not None and len(ref_paths) > 0:
            score = correspondence_score(
                out,
                ref_paths,
                binary_mask_image,
                args.match_threshold,
                loftr,
                device,
            )

        candidates.append({
            "candidate_idx": i,
            "score": score,
            "image": out,
        })

    # Rank candidates if scores exist
    if loftr is not None and len(ref_paths) > 0:
        candidates = sorted(
            candidates,
            key=lambda x: x["score"] if x["score"] is not None else -1,
            reverse=True,
        )

    top_k = args.top_k if args.top_k is not None else len(candidates)
    kept = candidates[:top_k]

    # Save ranked outputs
    for rank, item in enumerate(kept):
        item["image"].save(os.path.join(args.output_dir, f"{rank:02d}.png"))

    # Save scores JSON like the initial program
    if args.scores_json is not None:
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

    print(f"\nDone. Images saved to: {args.output_dir}")
    if args.scores_json is not None:
        print(f"Scores JSON saved to: {args.scores_json}")

    print("\n── CBI configuration ──")
    print(f"  n_rings          = {n_rings}")
    print(f"  ring_kernel_size = {args.ring_kernel_size} px")
    print(f"  steps per pass   = {args.num_inference_steps}")
    print(f"  total pipeline calls = {n_rings} × {args.num_images} = {n_rings * args.num_images}")
    print(f"  time overhead vs one-shot ≈ {n_rings}×")


if __name__ == "__main__":
    main()


"""
python infer_ring.py `
  --model_dir="bench0-model" `
  --validation_image="realfill_dataset/RealBench/0/target/target.png" `
  --validation_mask="realfill_dataset/RealBench/0/target/mask.png" `
  --reference_dir="realfill_dataset/RealBench/0/ref" `
  --output_dir="bench0-ring-16ranked" `
  --num_images=16 `
  --top_k=16 `
  --scores_json="bench0-ring-16ranked/scores.json" `
  --num_inference_steps=50 `
  --guidance_scale=1.0 `
  --resolution=512 `
  --n_rings=4 `
  --ring_kernel_size=24 `
  --seed=42
"""
