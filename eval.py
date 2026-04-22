import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTS = {".png", ".jpg", ".jpeg"}

_lpips_fn = None
_clip_model = None
_clip_preprocess = None
_dreamsim_model = None
_dreamsim_preprocess = None
_dino_processor = None
_dino_model = None


def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    return _lpips_fn


def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None or _clip_preprocess is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def get_dreamsim():
    global _dreamsim_model, _dreamsim_preprocess
    if _dreamsim_model is None or _dreamsim_preprocess is None:
        from dreamsim import dreamsim
        dreamsim_cache = os.path.join(os.path.expanduser("~"), ".cache", "dreamsim")
        os.makedirs(dreamsim_cache, exist_ok=True)
        _dreamsim_model, _dreamsim_preprocess = dreamsim(pretrained=True, cache_dir=dreamsim_cache)
        _dreamsim_model = _dreamsim_model.to(device)
        _dreamsim_model.eval()
    return _dreamsim_model, _dreamsim_preprocess


def get_dino():
    global _dino_processor, _dino_model
    if _dino_processor is None or _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel
        dino_model_name = "facebook/dinov2-base"
        _dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        _dino_model = AutoModel.from_pretrained(dino_model_name).to(device).eval()
    return _dino_processor, _dino_model

def get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2

def load_image_pil(path):
    return Image.open(path).convert("RGB")


def resize_pil_if_needed(img, size, is_mask=False):
    """
    size is a PIL-style size tuple: (width, height)
    """
    if img.size == size:
        return img
    interp = Image.NEAREST if is_mask else Image.LANCZOS
    return img.resize(size, interp)


def prepare_lowlevel_inputs(gen_path, ref_path, mask_path, lowlevel_size=None):
    """
    Prepare aligned low-level inputs for PSNR / SSIM / LPIPS.

    If lowlevel_size is None:
        use reference image size.
    Else:
        use a fixed square size (lowlevel_size, lowlevel_size).
    """
    ref_pil = load_image_pil(ref_path)
    gen_pil = load_image_pil(gen_path)
    mask_pil = Image.open(mask_path).convert("L")

    if lowlevel_size is None:
        target_size = ref_pil.size   # (W, H)
    else:
        target_size = (lowlevel_size, lowlevel_size)

    ref_pil = resize_pil_if_needed(ref_pil, target_size, is_mask=False)
    gen_pil = resize_pil_if_needed(gen_pil, target_size, is_mask=False)
    mask_pil = resize_pil_if_needed(mask_pil, target_size, is_mask=True)

    ref_np = np.array(ref_pil)                      # HWC uint8 RGB
    gen_np = np.array(gen_pil)                      # HWC uint8 RGB
    mask_np = (np.array(mask_pil) > 127).astype(np.uint8)   # HW, {0,1}

    ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).float() / 255.0
    gen_tensor = torch.from_numpy(gen_np).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)

    ref_tensor = ref_tensor.unsqueeze(0).to(device)   # [1,3,H,W]
    gen_tensor = gen_tensor.unsqueeze(0).to(device)   # [1,3,H,W]
    mask_tensor = mask_tensor.unsqueeze(0).to(device) # [1,1,H,W]

    return {
        "ref_np": ref_np,
        "gen_np": gen_np,
        "mask_np": mask_np,
        "ref_tensor": ref_tensor,
        "gen_tensor": gen_tensor,
        "mask_tensor": mask_tensor,
    }


def compute_masked_lpips(ref_tensor, gen_tensor, mask_tensor):
    """
    ref_tensor/gen_tensor: [1,3,H,W] in [0,1]
    mask_tensor: [1,1,H,W] in {0,1}
    """
    lpips_fn = get_lpips()

    mask3 = mask_tensor.repeat(1, ref_tensor.shape[1], 1, 1)

    # First normalize to [-1, 1], then mask.
    ref_norm = ref_tensor * 2.0 - 1.0
    gen_norm = gen_tensor * 2.0 - 1.0

    ref_masked = ref_norm * mask3
    gen_masked = gen_norm * mask3

    with torch.no_grad():
        return lpips_fn(ref_masked, gen_masked).item()


def compute_masked_psnr(ref_np, gen_np, mask_np):
    """
    ref_np/gen_np: HWC uint8 RGB
    mask_np: HW uint8 in {0,1}
    """
    if ref_np.shape != gen_np.shape or ref_np.shape[:2] != mask_np.shape[:2]:
        return float("nan")

    mask_bool = mask_np > 0
    if not np.any(mask_bool):
        return float("nan")

    ref_masked = ref_np[mask_bool]
    gen_masked = gen_np[mask_bool]

    if ref_masked.size == 0:
        return float("nan")

    mse = np.mean((ref_masked.astype(np.float64) - gen_masked.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")

    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def _ssim_single_channel_masked(img1_ch, img2_ch, mask_np, C1=(0.01 * 255) ** 2, C2=(0.03 * 255) ** 2):
    """
    img1_ch/img2_ch: HW uint8 or float images
    mask_np: HW uint8 in {0,1}
    """
    img1 = img1_ch.astype(np.float64)
    img2 = img2_ch.astype(np.float64)
    mask_bool = mask_np > 0

    if not np.any(mask_bool):
        return 0.0

    kernel_size = 11
    sigma = 1.5
    window = cv2.getGaussianKernel(kernel_size, sigma)
    window = np.outer(window, window.transpose())

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    sigma1_sq = np.maximum(0.0, sigma1_sq)
    sigma2_sq = np.maximum(0.0, sigma2_sq)

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    ssim_values = ssim_map[mask_bool]
    if ssim_values.size == 0:
        return 0.0

    mean_ssim = np.mean(ssim_values)
    return float(np.clip(mean_ssim, 0.0, 1.0))


def compute_masked_ssim(ref_np, gen_np, mask_np):
    """
    ref_np/gen_np: HWC uint8 RGB
    mask_np: HW uint8 in {0,1}
    """
    if ref_np.shape != gen_np.shape or ref_np.shape[:2] != mask_np.shape[:2]:
        return float("nan")

    if ref_np.ndim == 2:
        return _ssim_single_channel_masked(ref_np, gen_np, mask_np)

    if ref_np.ndim == 3 and ref_np.shape[2] == 3:
        ssims = []
        for c in range(3):
            ssims.append(_ssim_single_channel_masked(ref_np[:, :, c], gen_np[:, :, c], mask_np))
        return float(np.mean(ssims))

    return float("nan")


def compute_clip_similarity(img1_path, img2_path):
    clip_model, clip_preprocess = get_clip()
    img1 = clip_preprocess(load_image_pil(img1_path)).unsqueeze(0).to(device)
    img2 = clip_preprocess(load_image_pil(img2_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        f1 = clip_model.encode_image(img1)
        f2 = clip_model.encode_image(img2)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return torch.cosine_similarity(f1, f2).item()


def _ensure_batch(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x


def compute_dreamsim(img1_path, img2_path):
    dreamsim_model, dreamsim_preprocess = get_dreamsim()
    img1 = dreamsim_preprocess(load_image_pil(img1_path)).to(device)
    img2 = dreamsim_preprocess(load_image_pil(img2_path)).to(device)
    img1 = _ensure_batch(img1)
    img2 = _ensure_batch(img2)

    with torch.no_grad():
        dist = dreamsim_model(img1, img2)

    if isinstance(dist, torch.Tensor):
        return float(dist.squeeze().detach().cpu().item())
    return float(dist)


def compute_dino_similarity(img1_path, img2_path):
    dino_processor, dino_model = get_dino()

    img1 = load_image_pil(img1_path)
    img2 = load_image_pil(img2_path)

    inputs1 = dino_processor(images=img1, return_tensors="pt").to(device)
    inputs2 = dino_processor(images=img2, return_tensors="pt").to(device)

    with torch.no_grad():
        out1 = dino_model(**inputs1)
        out2 = dino_model(**inputs2)

        emb1 = out1.last_hidden_state[:, 0]  # CLS token
        emb2 = out2.last_hidden_state[:, 0]

        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)

    return torch.cosine_similarity(emb1, emb2).item()


def list_images(folder):
    folder = Path(folder)
    return sorted([
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ])


def select_topk_images_from_scores(generated_dir, scores_json, top_k):
    """
    Select top-k image paths from generated_dir according to scores.json.

    Assumes infer.py saved ranked images as:
        00.png, 01.png, 02.png, ...
    where filename index is rank after sorting.
    """
    all_images = list_images(generated_dir)
    image_map = {Path(p).name: p for p in all_images}

    with open(scores_json, "r", encoding="utf-8") as f:
        score_data = json.load(f)

    if not isinstance(score_data, list):
        raise ValueError(f"scores_json must contain a list, got: {type(score_data)}")

    # If rank field exists, use it. Otherwise assume JSON order matches rank order.
    has_rank = all(isinstance(item, dict) and ("rank" in item) for item in score_data)

    selected = []
    if has_rank:
        ranked = sorted(score_data, key=lambda x: int(x["rank"]))
        for item in ranked[:top_k]:
            rank = int(item["rank"])
            fname = f"{rank:02d}.png"
            if fname not in image_map:
                raise ValueError(f"Could not find ranked image '{fname}' inside {generated_dir}")
            selected.append(image_map[fname])
    else:
        selected = all_images[:top_k]

    return selected


def summarize_topk_from_cached_eval(cached_eval_json, top_k):
    """
    Read a previously saved single-set eval JSON (with per_image metrics),
    and re-average only the first top_k entries.
    """
    with open(cached_eval_json, "r", encoding="utf-8") as f:
        cached = json.load(f)

    if "per_image" not in cached:
        raise ValueError(f"{cached_eval_json} does not contain 'per_image'.")

    per_image = cached["per_image"]
    if not isinstance(per_image, list) or len(per_image) == 0:
        raise ValueError(f"{cached_eval_json} has empty or invalid 'per_image'.")

    if top_k > len(per_image):
        raise ValueError(f"Requested top_k={top_k}, but cached eval has only {len(per_image)} entries.")

    selected = per_image[:top_k]
    metrics = ["PSNR", "SSIM", "LPIPS", "DreamSim", "DINO", "CLIP"]

    summary = {"num_images": len(selected)}
    for metric in metrics:
        values = [row[metric] for row in selected if metric in row]
        if len(values) == 0:
            raise ValueError(f"Metric '{metric}' not found in cached per_image entries.")
        summary[metric] = float(np.mean(values))

    return {
        "source_cached_eval_json": str(cached_eval_json),
        "reference_path": cached.get("reference_path", None),
        "generated_dir": cached.get("generated_dir", None),
        "top_k_from_cached": top_k,
        "summary": summary,
        "per_image": selected,
    }


def evaluate_one_set(generated_dir, reference_path, mask_path, scores_json=None, top_k_by_score=None, lowlevel_size=None):
    """
    Evaluate one folder of generated images.
    If scores_json + top_k_by_score are provided, evaluate only the top-k ranked images.
    """
    if scores_json is not None and top_k_by_score is not None:
        gen_images = select_topk_images_from_scores(generated_dir, scores_json, top_k_by_score)
    else:
        gen_images = list_images(generated_dir)

    if len(gen_images) == 0:
        raise ValueError(f"No images found in: {generated_dir}")

    per_image = []
    psnr_scores, ssim_scores = [], []
    lpips_scores, dreamsim_scores = [], []
    dino_scores, clip_scores = [], []

    for path in gen_images:
        low = prepare_lowlevel_inputs(
            gen_path=path,
            ref_path=reference_path,
            mask_path=mask_path,
            lowlevel_size=lowlevel_size,
        )

        result = {
            "image": os.path.basename(path),
            "PSNR": compute_masked_psnr(low["ref_np"], low["gen_np"], low["mask_np"]),
            "SSIM": compute_masked_ssim(low["ref_np"], low["gen_np"], low["mask_np"]),
            "LPIPS": compute_masked_lpips(low["ref_tensor"], low["gen_tensor"], low["mask_tensor"]),
            "DreamSim": compute_dreamsim(path, reference_path),
            "DINO": compute_dino_similarity(path, reference_path),
            "CLIP": compute_clip_similarity(path, reference_path),
        }

        psnr_scores.append(result["PSNR"])
        ssim_scores.append(result["SSIM"])
        lpips_scores.append(result["LPIPS"])
        dreamsim_scores.append(result["DreamSim"])
        dino_scores.append(result["DINO"])
        clip_scores.append(result["CLIP"])

        per_image.append(result)

    summary = {
        "num_images": len(gen_images),
        "PSNR": float(np.mean(psnr_scores)),
        "SSIM": float(np.mean(ssim_scores)),
        "LPIPS": float(np.mean(lpips_scores)),
        "DreamSim": float(np.mean(dreamsim_scores)),
        "DINO": float(np.mean(dino_scores)),
        "CLIP": float(np.mean(clip_scores)),
    }

    return {
        "generated_dir": str(generated_dir),
        "reference_path": str(reference_path),
        "scores_json": str(scores_json) if scores_json is not None else None,
        "top_k_by_score": top_k_by_score,
        "summary": summary,
        "per_image": per_image,
    }


def evaluate_multiscene(parent_dir, realbench_root, lowlevel_size=None):
    """
    Evaluate many RealBench scenes, where each subfolder name in parent_dir is the scene id.

    Example:
        parent_dir/
            0/
                00.png ... 15.png
            1/
                00.png ... 15.png
            2/
                00.png ... 15.png

    Ground truth / masks are read from:
        realbench_root/{scene_id}/target/gt.png
        realbench_root/{scene_id}/target/mask.png
    """
    parent = Path(parent_dir)
    realbench_root = Path(realbench_root)

    subdirs = sorted([p for p in parent.iterdir() if p.is_dir()])
    if len(subdirs) == 0:
        raise ValueError(f"No subdirectories found in: {parent_dir}")

    all_results = []

    for subdir in subdirs:
        scene_id = subdir.name

        reference_path = realbench_root / scene_id / "target" / "gt.png"
        mask_path = realbench_root / scene_id / "target" / "mask.png"

        if not reference_path.exists():
            raise ValueError(f"Ground-truth not found for scene {scene_id}: {reference_path}")
        if not mask_path.exists():
            raise ValueError(f"Mask not found for scene {scene_id}: {mask_path}")

        try:
            res = evaluate_one_set(subdir, reference_path, mask_path, lowlevel_size=lowlevel_size)
            res["scene_id"] = scene_id
            all_results.append(res)
        except ValueError:
            continue

    if len(all_results) == 0:
        raise ValueError(f"No valid scene folders found in: {parent_dir}")

    macro = {
        "num_scenes": len(all_results),
        "PSNR": float(np.mean([r["summary"]["PSNR"] for r in all_results])),
        "SSIM": float(np.mean([r["summary"]["SSIM"] for r in all_results])),
        "LPIPS": float(np.mean([r["summary"]["LPIPS"] for r in all_results])),
        "DreamSim": float(np.mean([r["summary"]["DreamSim"] for r in all_results])),
        "DINO": float(np.mean([r["summary"]["DINO"] for r in all_results])),
        "CLIP": float(np.mean([r["summary"]["CLIP"] for r in all_results])),
    }

    return {
        "parent_dir": str(parent_dir),
        "realbench_root": str(realbench_root),
        "macro_average_over_scenes": macro,
        "per_scene": [
            {
                "scene_id": r["scene_id"],
                **r["summary"]
            }
            for r in all_results
        ],
        "full_results": all_results,
    }


def print_summary_single(result):
    s = result["summary"]
    print(f"Evaluated set: {result['generated_dir']}")
    print(f"Reference:     {result['reference_path']}")
    print(f"Num images:    {s['num_images']}")
    print("-" * 40)
    print(f"PSNR      ↑  {s['PSNR']:.4f}")
    print(f"SSIM      ↑  {s['SSIM']:.4f}")
    print(f"LPIPS     ↓  {s['LPIPS']:.4f}")
    print(f"DreamSim  ↓  {s['DreamSim']:.4f}")
    print(f"DINO      ↑  {s['DINO']:.4f}")
    print(f"CLIP      ↑  {s['CLIP']:.4f}")


def print_summary_multiscene(result):
    m = result["macro_average_over_scenes"]
    print(f"Evaluated parent dir: {result['parent_dir']}")
    print(f"RealBench root:       {result['realbench_root']}")
    print(f"Num scenes:           {m['num_scenes']}")
    print("-" * 40)
    print("Macro average over scenes:")
    print(f"PSNR      ↑  {m['PSNR']:.4f}")
    print(f"SSIM      ↑  {m['SSIM']:.4f}")
    print(f"LPIPS     ↓  {m['LPIPS']:.4f}")
    print(f"DreamSim  ↓  {m['DreamSim']:.4f}")
    print(f"DINO      ↑  {m['DINO']:.4f}")
    print(f"CLIP      ↑  {m['CLIP']:.4f}")
    print("-" * 40)
    print("Per-scene summary:")
    for row in result["per_scene"]:
        print(
            f"Scene {row['scene_id']}: "
            f"PSNR={row['PSNR']:.4f}, "
            f"SSIM={row['SSIM']:.4f}, "
            f"LPIPS={row['LPIPS']:.4f}, "
            f"DreamSim={row['DreamSim']:.4f}, "
            f"DINO={row['DINO']:.4f}, "
            f"CLIP={row['CLIP']:.4f}"
        )


def print_summary_cached_topk(result):
    s = result["summary"]
    print(f"Cached source: {result['source_cached_eval_json']}")
    print(f"Original dir:  {result.get('generated_dir', None)}")
    print(f"Reference:     {result.get('reference_path', None)}")
    print(f"Top-k used:    {result['top_k_from_cached']}")
    print(f"Num images:    {s['num_images']}")
    print("-" * 40)
    print(f"PSNR      ↑  {s['PSNR']:.4f}")
    print(f"SSIM      ↑  {s['SSIM']:.4f}")
    print(f"LPIPS     ↓  {s['LPIPS']:.4f}")
    print(f"DreamSim  ↓  {s['DreamSim']:.4f}")
    print(f"DINO      ↑  {s['DINO']:.4f}")
    print(f"CLIP      ↑  {s['CLIP']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated images with PSNR / SSIM / LPIPS / DreamSim / DINO / CLIP"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiscene"],
        default=None,
        help="single: evaluate one folder of generated images; multiscene: evaluate many RealBench scenes."
    )

    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the ground-truth image (required in single mode)."
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to mask.png defining the filled region (required in single mode)."
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default=None,
        help="Used in single mode: folder containing one set of generated images."
    )

    parser.add_argument(
        "--parent_dir",
        type=str,
        default=None,
        help="Used in multiscene mode: parent folder containing one subfolder per scene id."
    )
    parser.add_argument(
        "--realbench_root",
        type=str,
        default=None,
        help="Used in multiscene mode: root folder of RealBench, e.g. 'realfill_dataset/RealBench'."
    )

    parser.add_argument(
        "--scores_json",
        type=str,
        default=None,
        help="Optional scores.json from ranked inference. If used with --top_k_by_score, only top-k images are evaluated."
    )
    parser.add_argument(
        "--top_k_by_score",
        type=int,
        default=None,
        help="If used with --scores_json in single mode, evaluate only the top-k ranked images from generated_dir."
    )

    parser.add_argument(
        "--cached_eval_json",
        type=str,
        default=None,
        help="Path to a previously saved single-set evaluation JSON that already contains per_image metrics."
    )
    parser.add_argument(
        "--top_k_from_cached",
        type=int,
        default=None,
        help="If used with --cached_eval_json, re-average only the first top-k per-image entries from the cached JSON."
    )

    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save the evaluation result as JSON."
    )
    parser.add_argument(
        "--lowlevel_size",
        type=int,
        default=None,
        help="Optional fixed size for low-level metrics (PSNR/SSIM/LPIPS). If omitted, use ground-truth image size."
    )

    args = parser.parse_args()

    if args.cached_eval_json is not None:
        if args.top_k_from_cached is None:
            raise ValueError("--top_k_from_cached is required when using --cached_eval_json")
        result = summarize_topk_from_cached_eval(args.cached_eval_json, args.top_k_from_cached)
        print_summary_cached_topk(result)

    elif args.mode == "single":
        if args.generated_dir is None:
            raise ValueError("--generated_dir is required in single mode")
        if args.reference_path is None:
            raise ValueError("--reference_path is required in single mode")
        if args.mask_path is None:
            raise ValueError("--mask_path is required in single mode")

        result = evaluate_one_set(
            args.generated_dir,
            args.reference_path,
            args.mask_path,
            scores_json=args.scores_json,
            top_k_by_score=args.top_k_by_score,
            lowlevel_size=args.lowlevel_size,
        )
        print_summary_single(result)

    elif args.mode == "multiscene":
        if args.parent_dir is None:
            raise ValueError("--parent_dir is required in multiscene mode")
        if args.realbench_root is None:
            raise ValueError("--realbench_root is required in multiscene mode")

        result = evaluate_multiscene(
            args.parent_dir,
            args.realbench_root,
            lowlevel_size=args.lowlevel_size,
        )
        print_summary_multiscene(result)

    else:
        raise ValueError("You must use either --cached_eval_json, or set --mode to single/multiscene.")

    if args.save_json is not None:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()


# python eval.py `
#   --mode=single `
#   --generated_dir="bench0-32ranked" `
#   --reference_path="realfill_dataset/RealBench/0/target/gt.png" `
#   --mask_path="realfill_dataset/RealBench/0/target/mask.png" `
#   --save_json="bench0_32ranked_eval.json"

# python eval.py `
#   --cached_eval_json="bench0_32ranked_eval.json" `
#   --top_k_from_cached=16 `
#   --save_json="bench0_top16_eval.json"

# python eval.py `
#   --mode=multiscene `
#   --parent_dir="all_images" `
#   --realbench_root="realfill_dataset/RealBench" `
#   --save_json="all_images_multiscene_eval.json"