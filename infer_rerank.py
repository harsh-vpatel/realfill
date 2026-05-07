import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import kornia as K
from kornia.feature import LoFTR


IMAGE_SIZE = 512
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOFTR_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerank existing LoFTR-filtered RealFill candidates using a non-learned structural-shortlist reranker."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing existing ranked images, e.g. bench4-32ranked_top16",
    )
    parser.add_argument(
        "--input_scores_json",
        type=str,
        default=None,
        help="Existing scores.json from the original LoFTR ranking.",
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        required=True,
        help="Path to target mask.png",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Reference image directory for optional DINO similarity and optional LoFTR recomputation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save reranked images and scores.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=16,
        help="How many reranked images to save. Default: 16",
    )

    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for LoFTR recomputation if needed.",
    )

    # Optional semantic / quality features
    parser.add_argument(
        "--use_dino_ref",
        action="store_true",
        help="Compute DINOv2 similarity to references if local weights are available.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vits14",
        help="torch.hub DINOv2 model name, e.g. dinov2_vits14 / dinov2_vitb14",
    )

    parser.add_argument(
        "--use_topiq_nr",
        action="store_true",
        help="Use TOPIQ-NR via local pyiqa if available.",
    )
    parser.add_argument(
        "--use_maniqa",
        action="store_true",
        help="Use MANIQA via local pyiqa if available.",
    )
    parser.add_argument(
        "--use_clipiqa",
        action="store_true",
        help="Use CLIP-IQA via local pyiqa if available.",
    )

    parser.add_argument(
        "--topiq_json",
        type=str,
        default=None,
        help="Optional precomputed TOPIQ-NR JSON.",
    )
    parser.add_argument(
        "--maniqa_json",
        type=str,
        default=None,
        help="Optional precomputed MANIQA JSON.",
    )
    parser.add_argument(
        "--clipiqa_json",
        type=str,
        default=None,
        help="Optional precomputed CLIP-IQA JSON.",
    )

    # Non-learned reranker hyperparameters
    parser.add_argument(
        "--shortlist_size",
        type=int,
        default=8,
        help="How many candidates to keep in the structural shortlist.",
    )
    parser.add_argument(
        "--bad_boundary_penalty",
        type=float,
        default=0.10,
        help="Penalty applied inside shortlist to very poor seam cases.",
    )

    return parser.parse_args()


def load_candidate_image_paths(input_dir: str) -> dict:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    candidate_paths = {}
    for path in input_path.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and path.stem.isdigit():
            candidate_paths[int(path.stem)] = str(path)

    if not candidate_paths:
        raise ValueError(f"No ranked images (00.png, 01.png, ...) found in {input_dir}")

    return dict(sorted(candidate_paths.items()))


def load_reference_paths(reference_dir: str) -> list:
    reference_path = Path(reference_dir)
    if not reference_path.exists():
        raise ValueError(f"reference_dir does not exist: {reference_dir}")

    paths = sorted([
        str(path) for path in reference_path.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ])

    if len(paths) == 0:
        raise ValueError(f"No reference images found in {reference_dir}")

    return paths


def load_original_scores(scores_json_path: str) -> dict:
    with open(scores_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("scores_json must be a list")

    score_map = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rank = int(row.get("rank", len(score_map)))
        score_map[rank] = {
            "rank": rank,
            "candidate_index": int(row.get("candidate_idx", -1)),
            "score": None if row.get("score") is None else float(row["score"]),
        }
    return score_map


def load_binary_mask(mask_path: str) -> np.ndarray:
    """
    Returns HxW uint8 mask with 255 inside fill region and 0 outside.
    """
    mask = Image.open(mask_path).convert("L")
    mask = mask.point(lambda pixel: 255 if pixel > 127 else 0)
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


def load_optional_score_json(path: str) -> dict:
    """
    Supports:
      {"00.png": 0.87}
      {"0": 0.87}
      [{"rank": 0, "score": 0.87}, ...]
      [{"image": "00.png", "score": 0.87}, ...]
    """
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out[str(key)] = float(value)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            if "rank" in row and "score" in row:
                out[str(int(row["rank"]))] = float(row["score"])
            elif "image" in row and "score" in row:
                out[str(row["image"])] = float(row["score"])
    return out


def lookup_optional_score(score_map: dict, base_rank: int, image_name: str):
    for key in [image_name, str(base_rank), str(base_rank).zfill(2)]:
        if key in score_map:
            return score_map[key]
    return None


def image_to_loftr_gray(image_pil: Image.Image) -> torch.Tensor:
    tensor = LOFTR_IMAGE_TRANSFORM(image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    return K.color.rgb_to_grayscale(tensor)


def mask_generated_region(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    black_image = Image.new("RGB", image_pil.size, (0, 0, 0))
    return Image.composite(image_pil, black_image, mask_pil)


def compute_reference_match_score(
    candidate_image: Image.Image,
    reference_paths: list,
    binary_mask_pil: Image.Image,
    confidence_threshold: float,
    loftr_model,
) -> float:
    candidate_tensor = image_to_loftr_gray(mask_generated_region(candidate_image, binary_mask_pil))
    total_matches = 0

    with torch.no_grad():
        for reference_path in reference_paths:
            reference_image = Image.open(reference_path).convert("RGB")
            reference_tensor = image_to_loftr_gray(reference_image)

            output = loftr_model({"image0": candidate_tensor, "image1": reference_tensor})

            if "confidence" in output:
                total_matches += int((output["confidence"] > confidence_threshold).sum().item())
            else:
                total_matches += int(output["keypoints0"].shape[0])

    return float(total_matches)



def compute_stack_consensus_scores(candidate_stack: np.ndarray, fill_mask: np.ndarray) -> list:
    """
    How close each candidate is to the pixel-wise median image inside the fill region.
    Higher = closer to stack consensus.
    """
    if not fill_mask.any():
        return [0.0] * len(candidate_stack)

    stack_median = np.median(candidate_stack, axis=0)
    scores = []

    for candidate_image in candidate_stack:
        distance = np.mean(
            np.abs(
                candidate_image[fill_mask].astype(np.float32)
                - stack_median[fill_mask].astype(np.float32)
            )
        )
        scores.append(-float(distance))

    return scores


def compute_boundary_seam_scores(candidate_stack: np.ndarray, fill_mask: np.ndarray, band_width: int = 5) -> list:
    """
    Boundary seam score: mean color discontinuity across inner/outer mask bands.
    Lower seam -> higher score.
    """
    fill_mask_u8 = fill_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * band_width + 1, 2 * band_width + 1),
    )

    eroded_mask = cv2.erode(fill_mask_u8, kernel, iterations=1).astype(bool)
    dilated_mask = cv2.dilate(fill_mask_u8, kernel, iterations=1).astype(bool)

    inner_band = fill_mask & (~eroded_mask)
    outer_band = dilated_mask & (~fill_mask)

    if not inner_band.any() or not outer_band.any():
        return [0.0] * len(candidate_stack)

    scores = []
    for candidate_image in candidate_stack:
        blurred = cv2.GaussianBlur(candidate_image.astype(np.float32), (0, 0), sigmaX=1.0)
        seam_value = np.mean(np.abs(blurred[inner_band].mean(0) - blurred[outer_band].mean(0)))
        scores.append(-float(seam_value))

    return scores


def try_load_dino_model(model_name: str):
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        return model.to(DEVICE).eval()
    except Exception as e:
        print(f"[WARN] Could not load DINOv2 '{model_name}': {e}")
        return None


def compute_dino_embedding(image_pil: Image.Image, dino_model):
    tensor = DINO_TRANSFORM(image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = dino_model(tensor)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        feature = feature.flatten(1)
        feature = torch.nn.functional.normalize(feature, dim=1)
    return feature


def compute_dino_reference_similarity(candidate_images: list, reference_paths: list, dino_model) -> list:
    if dino_model is None or len(reference_paths) == 0:
        return [None] * len(candidate_images)

    reference_features = []
    for reference_path in reference_paths:
        reference_image = Image.open(reference_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        reference_features.append(compute_dino_embedding(reference_image, dino_model))
    reference_features = torch.cat(reference_features, dim=0)

    scores = []
    with torch.no_grad():
        for candidate_image in candidate_images:
            candidate_feature = compute_dino_embedding(
                candidate_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS),
                dino_model,
            )
            similarity = torch.mm(candidate_feature, reference_features.T).mean().item()
            scores.append(float(similarity))

    return scores


def try_import_pyiqa():
    try:
        import pyiqa
        return pyiqa
    except Exception as e:
        print(f"[WARN] pyiqa unavailable: {e}")
        return None


def build_pyiqa_metric(metric_name: str, pyiqa_module):
    try:
        return pyiqa_module.create_metric(metric_name, device=DEVICE)
    except Exception as e:
        print(f"[WARN] Could not create pyiqa metric '{metric_name}': {e}")
        return None


def compute_pyiqa_metric_scores(candidate_images: list, metric) -> list:
    if metric is None:
        return [None] * len(candidate_images)

    scores = []
    with torch.no_grad():
        for candidate_image in candidate_images:
            tensor = LOFTR_IMAGE_TRANSFORM(candidate_image.convert("RGB")).unsqueeze(0).to(DEVICE)
            output = metric(tensor)
            if isinstance(output, torch.Tensor):
                scores.append(float(output.squeeze().item()))
            else:
                scores.append(float(output))
    return scores


def robust_zscore(values, lower_quantile=0.10, upper_quantile=0.90):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values

    lower = float(np.quantile(values, lower_quantile))
    upper = float(np.quantile(values, upper_quantile))
    clipped = np.clip(values, lower, upper)

    mean = float(clipped.mean())
    std = float(clipped.std())
    if std < 1e-8:
        return np.zeros_like(clipped, dtype=np.float32)

    return (clipped - mean) / std


def rerank_with_structural_shortlist(
    candidates: list,
    shortlist_size: int = 8,
    bad_boundary_penalty: float = 0.10,
):
    """
    Stage 1: shortlist by structural reliability
      - reference_match_score
      - boundary_seam_score
      - stack_consensus_score

    Stage 2: rerank inside shortlist using softer semantic / perceptual cues
      - dino_reference_similarity
      - topiq_score
      - maniqa_score
      - clipiqa_score
    """

    def get_feature_array(feature_name):
        values = []
        for candidate in candidates:
            value = candidate.get(feature_name, None)
            values.append(0.0 if value is None else float(value))
        return np.asarray(values, dtype=np.float32)

    # Primary structural features
    reference_match_z = robust_zscore(get_feature_array("reference_match_score"))
    boundary_seam_z = robust_zscore(get_feature_array("boundary_seam_score"))
    stack_consensus_z = robust_zscore(get_feature_array("stack_consensus_score"))

    structural_score = (
        0.50 * reference_match_z
        + 0.25 * boundary_seam_z
        + 0.25 * stack_consensus_z
    )

    structural_order = np.argsort(-structural_score)
    shortlist_indices = structural_order[:min(shortlist_size, len(candidates))]

    # Secondary soft features
    dino_similarity_z = robust_zscore(get_feature_array("dino_reference_similarity"))
    topiq_z = robust_zscore(get_feature_array("topiq_score"))
    maniqa_z = robust_zscore(get_feature_array("maniqa_score"))
    clipiqa_z = robust_zscore(get_feature_array("clipiqa_score"))

    rerank_score = structural_score.copy()

    rerank_score[shortlist_indices] = (
        structural_score[shortlist_indices]
        + 0.10 * dino_similarity_z[shortlist_indices]
        + 0.05 * topiq_z[shortlist_indices]
        + 0.03 * maniqa_z[shortlist_indices]
        + 0.02 * clipiqa_z[shortlist_indices]
    )

    # Penalize worst seam cases inside shortlist
    raw_boundary = get_feature_array("boundary_seam_score")
    bad_seam_threshold = np.quantile(raw_boundary, 0.15)
    for idx in shortlist_indices:
        if raw_boundary[idx] <= bad_seam_threshold:
            rerank_score[idx] -= bad_boundary_penalty

    debug = {
        "mode": "revised_structural_shortlist_fallback",
        "shortlist_size": int(len(shortlist_indices)),
        "bad_boundary_penalty": float(bad_boundary_penalty),
        "structural_weights": {
            "reference_match_score": 0.50,
            "boundary_seam_score": 0.25,
            "stack_consensus_score": 0.25,
        },
        "secondary_weights": {
            "dino_reference_similarity": 0.10,
            "topiq_score": 0.05,
            "maniqa_score": 0.03,
            "clipiqa_score": 0.02,
        },
        "shortlist_indices": shortlist_indices.tolist(),
        "structural_score": structural_score.tolist(),
        "rerank_score": rerank_score.tolist(),
    }

    return rerank_score.tolist(), debug



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    candidate_image_paths = load_candidate_image_paths(args.input_dir)
    base_ranks = list(candidate_image_paths.keys())

    mask_array = load_binary_mask(args.validation_mask)
    fill_mask = mask_array > 127
    mask_pil = Image.fromarray(mask_array)

    original_score_map = {}
    if args.input_scores_json is not None:
        original_score_map = load_original_scores(args.input_scores_json)

    reference_paths = []
    if args.reference_dir:
        reference_paths = load_reference_paths(args.reference_dir)

    print(f"Loading {len(base_ranks)} candidates from {args.input_dir} ...")

    candidate_records = []
    candidate_stack = []
    candidate_images = []

    for base_rank in base_ranks:
        image_pil = (
            Image.open(candidate_image_paths[base_rank])
            .convert("RGB")
            .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        )
        image_np = np.array(image_pil, dtype=np.float32)

        reference_match_score = None
        candidate_index = base_rank

        if base_rank in original_score_map:
            reference_match_score = original_score_map[base_rank]["score"]
            candidate_index = original_score_map[base_rank]["candidate_index"]

        candidate_records.append({
            "base_rank": base_rank,
            "candidate_index": candidate_index,
            "image_name": f"{base_rank:02d}.png",
            "image": image_pil,
            "reference_match_score": None if reference_match_score is None else float(reference_match_score),
        })

        candidate_stack.append(image_np)
        candidate_images.append(image_pil)

    candidate_stack = np.stack(candidate_stack, axis=0)

    need_loftr_recompute = any(c["reference_match_score"] is None for c in candidate_records) and len(reference_paths) > 0
    if need_loftr_recompute:
        print("Recomputing LoFTR scores (no input_scores_json provided)...")
        loftr_model = LoFTR(pretrained="outdoor").to(DEVICE).eval()

        for candidate in candidate_records:
            if candidate["reference_match_score"] is None:
                candidate["reference_match_score"] = compute_reference_match_score(
                    candidate["image"],
                    reference_paths,
                    mask_pil,
                    args.match_threshold,
                    loftr_model,
                )

        del loftr_model
        torch.cuda.empty_cache()

    print("Computing features...")

    # F1: LoFTR / reference consistency
    reference_match_scores = [c["reference_match_score"] for c in candidate_records]
    print(f"  reference_match_score:     {'available' if any(v is not None for v in reference_match_scores) else 'MISSING'}")

    # F2: stack consensus
    stack_consensus_scores = compute_stack_consensus_scores(candidate_stack, fill_mask)
    print("  stack_consensus_score:     computed")

    # F3: boundary seam
    boundary_seam_scores = compute_boundary_seam_scores(candidate_stack, fill_mask, band_width=5)
    print("  boundary_seam_score:       computed")

    # F4: DINO similarity to references
    dino_model = None
    if args.use_dino_ref:
        dino_model = try_load_dino_model(args.dino_model_name)
    dino_reference_similarities = compute_dino_reference_similarity(candidate_images, reference_paths, dino_model)
    print(f"  dino_reference_similarity: {'computed' if any(v is not None for v in dino_reference_similarities) else 'skipped'}")

    # F5/F6/F7: NR-IQA scores
    topiq_score_map = load_optional_score_json(args.topiq_json)
    maniqa_score_map = load_optional_score_json(args.maniqa_json)
    clipiqa_score_map = load_optional_score_json(args.clipiqa_json)

    pyiqa_module = None
    if args.use_topiq_nr or args.use_maniqa or args.use_clipiqa:
        pyiqa_module = try_import_pyiqa()

    topiq_metric = build_pyiqa_metric("topiq_nr", pyiqa_module) if args.use_topiq_nr and pyiqa_module else None
    maniqa_metric = build_pyiqa_metric("maniqa", pyiqa_module) if args.use_maniqa and pyiqa_module else None
    clipiqa_metric = build_pyiqa_metric("clipiqa", pyiqa_module) if args.use_clipiqa and pyiqa_module else None

    topiq_scores_live = compute_pyiqa_metric_scores(candidate_images, topiq_metric)
    maniqa_scores_live = compute_pyiqa_metric_scores(candidate_images, maniqa_metric)
    clipiqa_scores_live = compute_pyiqa_metric_scores(candidate_images, clipiqa_metric)

    topiq_scores = []
    maniqa_scores = []
    clipiqa_scores = []

    for index, candidate in enumerate(candidate_records):
        image_name = candidate["image_name"]
        base_rank = candidate["base_rank"]

        topiq_precomputed = lookup_optional_score(topiq_score_map, base_rank, image_name)
        maniqa_precomputed = lookup_optional_score(maniqa_score_map, base_rank, image_name)
        clipiqa_precomputed = lookup_optional_score(clipiqa_score_map, base_rank, image_name)

        topiq_scores.append(float(topiq_precomputed) if topiq_precomputed is not None else topiq_scores_live[index])
        maniqa_scores.append(float(maniqa_precomputed) if maniqa_precomputed is not None else maniqa_scores_live[index])
        clipiqa_scores.append(float(clipiqa_precomputed) if clipiqa_precomputed is not None else clipiqa_scores_live[index])

    availability = lambda scores: "computed" if any(v is not None for v in scores) else "skipped"
    print(f"  topiq_score:               {availability(topiq_scores)}")
    print(f"  maniqa_score:              {availability(maniqa_scores)}")
    print(f"  clipiqa_score:             {availability(clipiqa_scores)}")

    for i, candidate in enumerate(candidate_records):
        candidate["stack_consensus_score"] = stack_consensus_scores[i]
        candidate["boundary_seam_score"] = boundary_seam_scores[i]
        candidate["dino_reference_similarity"] = dino_reference_similarities[i]
        candidate["topiq_score"] = topiq_scores[i]
        candidate["maniqa_score"] = maniqa_scores[i]
        candidate["clipiqa_score"] = clipiqa_scores[i]

    print("\nRunning revised structural-shortlist reranking...")

    rerank_scores, rerank_metadata = rerank_with_structural_shortlist(
        candidate_records,
        shortlist_size=args.shortlist_size,
        bad_boundary_penalty=args.bad_boundary_penalty,
    )

    for i, candidate in enumerate(candidate_records):
        candidate["rerank_score"] = float(rerank_scores[i])

    candidate_records.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
    selected_records = candidate_records[:min(args.top_k, len(candidate_records))]

    for new_rank, candidate in enumerate(selected_records):
        candidate["image"].save(os.path.join(args.output_dir, f"{new_rank:02d}.png"))

    output_rows = []
    for new_rank, candidate in enumerate(selected_records):
        output_rows.append({
            "rank": new_rank,
            "base_rank": candidate["base_rank"],
            "candidate_index": candidate["candidate_index"],
            "score": candidate["rerank_score"],
            "rerank_score": candidate["rerank_score"],
            "reference_match_score": candidate["reference_match_score"],
            "dino_reference_similarity": candidate["dino_reference_similarity"],
            "topiq_score": candidate["topiq_score"],
            "maniqa_score": candidate["maniqa_score"],
            "clipiqa_score": candidate["clipiqa_score"],
            "boundary_seam_score": candidate["boundary_seam_score"],
            "stack_consensus_score": candidate["stack_consensus_score"],
            "ranking_mode": rerank_metadata["mode"],
        })

    scores_output_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_output_path, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)

    feature_dump = {
        "input_dir": args.input_dir,
        "reference_dir": args.reference_dir,
        "rerank_metadata": rerank_metadata,
        "all_candidates": [
            {
                "base_rank": candidate["base_rank"],
                "candidate_index": candidate["candidate_index"],
                "image_name": candidate["image_name"],
                "reference_match_score": candidate["reference_match_score"],
                "dino_reference_similarity": candidate["dino_reference_similarity"],
                "topiq_score": candidate["topiq_score"],
                "maniqa_score": candidate["maniqa_score"],
                "clipiqa_score": candidate["clipiqa_score"],
                "boundary_seam_score": candidate["boundary_seam_score"],
                "stack_consensus_score": candidate["stack_consensus_score"],
                "rerank_score": candidate["rerank_score"],
            }
            for candidate in candidate_records
        ],
    }

    feature_dump_path = os.path.join(args.output_dir, "rerank_features_full.json")
    with open(feature_dump_path, "w", encoding="utf-8") as f:
        json.dump(feature_dump, f, indent=2)

    print("\n── Results ──")
    print(f"  Saved top {len(selected_records)} reranked images to: {args.output_dir}")
    print(f"  Reranking mode: {rerank_metadata['mode']}")
    print("  Top 5 after reranking:")

    for candidate in selected_records[:5]:
        print(
            f"    base_rank={candidate['base_rank']:02d}  "
            f"rerank_score={candidate['rerank_score']:.5f}  "
            f"reference_match={candidate['reference_match_score']}  "
            f"boundary={candidate['boundary_seam_score']:.5f}  "
            f"consensus={candidate['stack_consensus_score']:.5f}"
        )


"""
python infer_rerank.py `
  --input_dir="bench31-32ranked_top16_ramr" `
  --input_scores_json="bench31-32ranked_top16_ramr/scores.json" `
  --validation_mask="realfill_dataset/RealBench/31/target/mask.png" `
  --reference_dir="realfill_dataset/RealBench/31/ref" `
  --output_dir="bench31-32ranked_top16_ramr" `
  --top_k=16 `
  --use_dino_ref `
  --dino_model_name="dinov2_vits14" `
  --use_topiq_nr `
  --use_maniqa `
  --use_clipiqa `
  --shortlist_size=8 `
  --bad_boundary_penalty=0.10
"""
