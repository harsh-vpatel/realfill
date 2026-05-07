import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline

from torchvision import transforms
import kornia as K
from kornia.feature import LoFTR


RESOLUTION = 512
PROMPT = "a photo of sks"
GUIDANCE_SCALE = 1.0
SECOND_STAGE_SCHEDULER = "ddpm"   
UNCERTAINTY_TOP_K = 4             # use top 4 strong candidates for uncertainty

VALID_EXTS = {".png", ".jpg", ".jpeg"}

_MATCH_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def pil_to_loftr_gray(pil_img: Image.Image, device: str) -> torch.Tensor:
    t = _MATCH_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)
    return K.color.rgb_to_grayscale(t)


def masked_candidate_for_matching(gen_img: Image.Image, binary_mask_img: Image.Image) -> Image.Image:
    black = Image.new("RGB", gen_img.size, (0, 0, 0))
    return Image.composite(gen_img, black, binary_mask_img)


def correspondence_score(
    gen_img: Image.Image,
    ref_paths: list[str],
    binary_mask_img: Image.Image,
    threshold: float,
    loftr: LoFTR,
    device: str,
) -> int:
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


def feather_mask(mask_pil: Image.Image, radius: int = 3) -> Image.Image:
    if radius <= 0:
        return mask_pil
    return mask_pil.filter(ImageFilter.BoxBlur(radius))


def load_binary_mask(mask_path: str) -> Image.Image:
    binary_mask_image = Image.open(mask_path).convert("L")
    binary_mask_image = binary_mask_image.point(lambda p: 255 if p > 127 else 0)
    binary_mask_image = binary_mask_image.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
    return binary_mask_image



def parse_scores_json(scores_json_path: str):
    """
    Parse scores.json produced by the updated RAMR infer.py.
    Expected keys per row:
      rank, candidate_idx, score
    and optionally:
      multi_cue_score, ref_score, consensus_score, sharpness_score, boundary_score
    """
    with open(scores_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("scores_json must contain a list.")

    parsed = []
    for row in data:
        if not isinstance(row, dict):
            continue
        parsed.append({
            "rank": int(row.get("rank", len(parsed))),
            "candidate_idx": int(row.get("candidate_idx", -1)),
            "score": None if row.get("score", None) is None else float(row["score"]),
            "multi_cue_score": None if row.get("multi_cue_score", None) is None else float(row["multi_cue_score"]),
            "ref_score": None if row.get("ref_score", None) is None else float(row["ref_score"]),
            "consensus_score": None if row.get("consensus_score", None) is None else float(row["consensus_score"]),
            "sharpness_score": None if row.get("sharpness_score", None) is None else float(row["sharpness_score"]),
            "boundary_score": None if row.get("boundary_score", None) is None else float(row["boundary_score"]),
        })

    if len(parsed) == 0:
        raise ValueError("scores_json did not contain valid ranking entries.")

    parsed = sorted(parsed, key=lambda x: x["rank"])
    return parsed


def load_ranked_paths(ranked_images_dir: str):
    ranked_dir = Path(ranked_images_dir)
    if not ranked_dir.exists():
        raise ValueError(f"ranked_images_dir does not exist: {ranked_images_dir}")

    ranked_paths = {}
    for p in ranked_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        stem = p.stem
        if stem.isdigit():
            ranked_paths[int(stem)] = str(p)

    if len(ranked_paths) == 0:
        raise ValueError(
            f"No ranked images like 00.png, 01.png, ... found in {ranked_images_dir}"
        )

    return ranked_paths


def load_image_for_rank(ranked_paths: dict, rank: int) -> Image.Image:
    if rank not in ranked_paths:
        raise ValueError(f"Rank {rank} not found in ranked_paths.")
    img = Image.open(ranked_paths[rank]).convert("RGB")
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    return img


def load_stack_for_ranks(ranked_paths: dict, ranks: list[int]) -> np.ndarray:
    imgs = []
    for rank in ranks:
        if rank not in ranked_paths:
            raise ValueError(f"Rank {rank} not found in ranked_paths.")
        img = Image.open(ranked_paths[rank]).convert("RGB")
        img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        imgs.append(np.array(img, dtype=np.float32))
    return np.stack(imgs, axis=0)



def compute_uncertainty_map(stack: np.ndarray, mask_np: np.ndarray, method: str = "mad") -> np.ndarray:
    """
    Compute per-pixel uncertainty inside the mask from candidate stack.

    method:
      - 'mad': Median Absolute Deviation (robust)
      - 'std': Standard deviation
    """
    if method == "mad":
        med = np.median(stack, axis=0)           # (N, H, W, 3) -> (H, W, 3)
        dev = np.abs(stack - med[None])          # (N, H, W, 3)
        mad = np.median(dev, axis=0)             # (H, W, 3)
        u = mad.mean(axis=-1)
    elif method == "std":
        std = np.std(stack, axis=0)
        u = std.mean(axis=-1)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    u = u.astype(np.float32)
    u[mask_np <= 127] = 0.0
    return u


def build_residual_mask(
    uncertainty_map: np.ndarray,
    mask_np: np.ndarray,
    quantile: float = 0.90,
    min_component_area: int = 64,
    morph_kernel: int = 5,
    dilate_px: int = 2,
) -> np.ndarray:
    vals = uncertainty_map[mask_np > 127]
    if vals.size == 0:
        return np.zeros_like(mask_np, dtype=np.uint8)

    thresh = np.quantile(vals, quantile)
    residual = ((uncertainty_map >= thresh) & (mask_np > 127)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel)

    if dilate_px > 0:
        dkernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        residual = cv2.dilate(residual, dkernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    cleaned = np.zeros_like(residual, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == lab] = 1

    return (cleaned * 255).astype(np.uint8)


def save_uncertainty_map(uncertainty_map: np.ndarray, save_path: str) -> None:
    if uncertainty_map.max() < 1e-6:
        Image.fromarray(np.zeros_like(uncertainty_map, dtype=np.uint8)).save(save_path)
        return
    vis = ((uncertainty_map / uncertainty_map.max()) * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat).save(save_path)



def set_scheduler(pipe: StableDiffusionInpaintPipeline):
    if SECOND_STAGE_SCHEDULER == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported hardcoded scheduler: {SECOND_STAGE_SCHEDULER}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAMR stage-2 residual repair using top-ranked outputs from updated infer.py."
    )

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to final exported RealFill model.")
    parser.add_argument("--validation_mask", type=str, required=True,
                        help="Path to mask.png")
    parser.add_argument("--ranked_images_dir", type=str, required=True,
                        help="Directory containing ranked outputs from updated infer.py "
                             "(00.png = best-ranked, 01.png = second-ranked, ...).")
    parser.add_argument("--scores_json", type=str, required=True,
                        help="scores.json from updated infer.py (top-k only).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save RAMR stage-2 outputs.")

    parser.add_argument("--residual_steps", type=int, default=25,
                        help="Inference steps for second-stage residual refinement.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for reproducibility.")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    parser.add_argument("--uncertainty_method", type=str, default="mad",
                        choices=["mad", "std"],
                        help="How to compute candidate disagreement.")
    parser.add_argument("--uncertainty_quantile", type=float, default=0.90,
                        help="Top uncertain quantile kept as residual mask.")
    parser.add_argument("--min_component_area", type=int, default=64,
                        help="Minimum connected-component area kept in residual mask.")
    parser.add_argument("--residual_dilate_px", type=int, default=2,
                        help="Residual mask dilation radius in pixels.")
    parser.add_argument("--residual_morph_kernel", type=int, default=5,
                        help="Morphology kernel size for residual mask cleanup.")
    parser.add_argument("--residual_mask_feather", type=int, default=2,
                        help="Feather radius for compositing residual refinement.")

    parser.add_argument("--reference_dir", type=str, default=None,
                        help="Optional reference image directory for LoFTR scoring of the final image.")
    parser.add_argument("--match_threshold", type=float, default=0.5,
                        help="Confidence threshold for counting LoFTR correspondences.")

    return parser.parse_args()



def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.mixed_precision]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model only for second-stage refinement
    print(f"Loading model from: {args.model_dir}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=weight_dtype,
        local_files_only=True,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load binary mask
    binary_mask_image = load_binary_mask(args.validation_mask)
    mask_np = np.array(binary_mask_image, dtype=np.uint8)
    mask_coverage = float((mask_np > 127).mean())
    print(f"Mask coverage: {mask_coverage:.1%} of image")

    # Load ranked metadata and image paths
    ranked_rows_all = parse_scores_json(args.scores_json)
    ranked_paths = load_ranked_paths(args.ranked_images_dir)
    available_ranks = set(ranked_paths.keys())

    # Keep only rows that actually exist as saved ranked images
    ranked_rows = [row for row in ranked_rows_all if row["rank"] in available_ranks]
    ranked_rows = sorted(ranked_rows, key=lambda x: x["rank"])

    if len(ranked_rows) == 0:
        raise ValueError("No overlapping ranks found between scores_json and ranked_images_dir.")

    expected_num = len(ranked_rows)

    # Optional LoFTR diagnostic for final image
    loftr = None
    ref_paths = []
    if args.reference_dir is not None:
        loftr = LoFTR(pretrained="outdoor").to(device).eval()
        ref_paths = sorted([
            str(p) for p in Path(args.reference_dir).iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ])
        print(f"Loaded {len(ref_paths)} reference images for final-image scoring")

    # Base image = ranked 00.png 
    best_rank_for_base = 0
    if best_rank_for_base not in ranked_paths:
        raise ValueError("ranked_images_dir must contain 00.png as the top-ranked base image.")

    ordered_ranks = [row["rank"] for row in ranked_rows]
    uncertainty_ranks_used = ordered_ranks[:min(UNCERTAINTY_TOP_K, len(ordered_ranks))]

    print(f"Base image source: ranked {best_rank_for_base:02d}.png")
    print(f"Top {len(uncertainty_ranks_used)} ranks used for uncertainty: {uncertainty_ranks_used}")

    # Load base image
    base_img = load_image_for_rank(ranked_paths, best_rank_for_base)
    base_img.save(os.path.join(args.output_dir, "00_base.png"))

    # Build uncertainty stack from top-K strong candidates
    uncertainty_stack = load_stack_for_ranks(ranked_paths, uncertainty_ranks_used)

    # Compute uncertainty map
    uncertainty_map = compute_uncertainty_map(
        stack=uncertainty_stack,
        mask_np=mask_np,
        method=args.uncertainty_method,
    )
    save_uncertainty_map(uncertainty_map, os.path.join(args.output_dir, "01_uncertainty_map.png"))

    # Build residual mask
    residual_mask_np = build_residual_mask(
        uncertainty_map=uncertainty_map,
        mask_np=mask_np,
        quantile=args.uncertainty_quantile,
        min_component_area=args.min_component_area,
        morph_kernel=args.residual_morph_kernel,
        dilate_px=args.residual_dilate_px,
    )
    residual_mask = Image.fromarray(residual_mask_np)
    residual_mask.save(os.path.join(args.output_dir, "02_residual_mask.png"))

    # Base row / score
    selected_base_row = None
    base_score = None
    for row in ranked_rows:
        if row["rank"] == best_rank_for_base:
            selected_base_row = row
            base_score = row["ref_score"]
            break

    final_score = None
    score_delta = None

    if (residual_mask_np > 127).sum() == 0:
        final = base_img
        final.save(os.path.join(args.output_dir, "04_final.png"))
        print(f"Residual mask is empty. Final output saved to: {args.output_dir}/04_final.png")

        if loftr is not None and len(ref_paths) > 0:
            final_score = correspondence_score(
                final,
                ref_paths,
                binary_mask_image,
                args.match_threshold,
                loftr,
                device,
            )
            print(f"Final image LoFTR score: {final_score}")

    else:
        set_scheduler(pipe)

        refine_generator = torch.Generator(device=device).manual_seed(args.seed + 999)

        refined = pipe(
            prompt=PROMPT,
            image=base_img,
            mask_image=residual_mask,
            num_inference_steps=args.residual_steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=refine_generator,
        ).images[0]

        feathered_residual_mask = feather_mask(residual_mask, args.residual_mask_feather)
        refined_partial = Image.composite(refined, base_img, feathered_residual_mask)
        refined_partial.save(os.path.join(args.output_dir, "03_refined_partial.png"))

        # Keep refined_partial directly to avoid re-compositing full mask
        final = refined_partial
        final.save(os.path.join(args.output_dir, "04_final.png"))
        print(f"Final output saved to: {args.output_dir}/04_final.png")

        if loftr is not None and len(ref_paths) > 0:
            final_score = correspondence_score(
                final,
                ref_paths,
                binary_mask_image,
                args.match_threshold,
                loftr,
                device,
            )
            print(f"Final refined image LoFTR score: {final_score}")

    if base_score is not None and final_score is not None:
        score_delta = final_score - base_score

    summary = {
        "ramr_mode": True,
        "best_rank_row": ranked_rows[0],
        "selected_base_row": selected_base_row,
        "num_ranked_candidates": expected_num,
        "resolution": RESOLUTION,
        "prompt": PROMPT,
        "guidance_scale": GUIDANCE_SCALE,
        "second_stage_scheduler": SECOND_STAGE_SCHEDULER,
        "residual_steps": args.residual_steps,
        "uncertainty_method": args.uncertainty_method,
        "uncertainty_quantile": args.uncertainty_quantile,
        "uncertainty_top_k": len(uncertainty_ranks_used),
        "uncertainty_ranks_used": uncertainty_ranks_used,
        "mask_coverage": mask_coverage,
        "residual_pixels": int((residual_mask_np > 127).sum()),
        "residual_fraction_of_image": float((residual_mask_np > 127).mean()),
        "base_score": base_score,
        "final_loftr_score": final_score,
        "score_delta": score_delta,
        "base_rank_selected": best_rank_for_base,
        "selected_base_image": f"{best_rank_for_base:02d}.png",
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  base image source        = ranked {best_rank_for_base:02d}.png")
    print(f"  selected candidate idx   = {selected_base_row['candidate_idx'] if selected_base_row else None}")
    print(f"  selected candidate score = {selected_base_row['score'] if selected_base_row else None}")
    print(f"  uncertainty top-k        = {len(uncertainty_ranks_used)}")
    print(f"  uncertainty ranks used   = {uncertainty_ranks_used}")
    print(f"  second-stage scheduler   = {SECOND_STAGE_SCHEDULER}")
    print(f"  residual_steps           = {args.residual_steps}")
    print(f"  uncertainty_method       = {args.uncertainty_method}")
    print(f"  uncertainty_quantile     = {args.uncertainty_quantile}")
    print(f"  min_component_area       = {args.min_component_area}")
    print(f"  residual_dilate_px       = {args.residual_dilate_px}")
    print(f"  residual_morph_kernel    = {args.residual_morph_kernel}")
    print(f"  residual_mask_feather    = {args.residual_mask_feather}")
    print(f"  mask coverage            = {mask_coverage:.1%}")
    print(f"  residual region size     = {(residual_mask_np > 127).mean():.1%} of image")
    print(f"  final LoFTR score        = {final_score}")
    print(f"  score delta vs base      = {score_delta}")


if __name__ == "__main__":
    main()


"""
python infer_residual_refilling.py `
  --model_dir="bench4-model" `
  --validation_mask="realfill_dataset/RealBench/4/target/mask.png" `
  --ranked_images_dir="bench4-32ranked_top16_ramr" `
  --scores_json="bench4-32ranked_top16_ramr/scores.json" `
  --reference_dir="realfill_dataset/RealBench/4/ref" `
  --output_dir="bench4-ramr2" `
  --residual_steps=25 `
  --uncertainty_method="mad" `
  --uncertainty_quantile=0.95 `
  --min_component_area=64 `
  --residual_dilate_px=2 `
  --residual_morph_kernel=5 `
  --residual_mask_feather=2 `
  --mixed_precision="fp16"
  --seed=42 `
"""
