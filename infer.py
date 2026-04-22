import argparse
import os
import json
from pathlib import Path
import torch
from PIL import Image, ImageFilter
import kornia as K
from torchvision import transforms
from kornia.feature import LoFTR
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
import numpy as np

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--validation_image",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation image",
)
parser.add_argument(
    "--validation_mask",
    type=str,
    default=None,
    required=True,
    help="The directory of the validation mask",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")

parser.add_argument(
    "--reference_dir",
    type=str,
    default=None,
    help="Directory containing reference images for LoFTR-based seed selection.",
)
parser.add_argument(
    "--num_candidates",
    type=int,
    default=64,
    help="Number of stochastic candidates to generate before LoFTR ranking.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=16,
    help="Number of top-ranked outputs to keep after LoFTR ranking.",
)
parser.add_argument(
    "--match_threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for counting LoFTR correspondences.",
)
parser.add_argument(
    "--scores_json",
    type=str,
    default=None,
    help="Optional path to save LoFTR correspondence scores as JSON.",
)


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTS = {".png", ".jpg", ".jpeg"}

match_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

loftr = LoFTR(pretrained="outdoor").to(device).eval()


def pil_to_loftr_gray(pil_img):
    t = match_transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    return K.color.rgb_to_grayscale(t)


def masked_candidate_for_matching(gen_img, binary_mask_img):
    black = Image.new("RGB", gen_img.size, (0, 0, 0))
    return Image.composite(gen_img, black, binary_mask_img)


def correspondence_score(gen_img, ref_paths, binary_mask_img, threshold):
    image0 = pil_to_loftr_gray(masked_candidate_for_matching(gen_img, binary_mask_img))
    total_matches = 0

    with torch.no_grad():
        for ref_path in ref_paths:
            ref_img = Image.open(ref_path).convert("RGB")
            image1 = pil_to_loftr_gray(ref_img)

            out = loftr({"image0": image0, "image1": image1})

            if "confidence" in out:
                total_matches += int((out["confidence"] > threshold).sum().item())
            else:
                total_matches += int(out["keypoints0"].shape[0])

    return total_matches

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    generator = None 

    # create & load model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        revision=None
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    ref_paths = []
    if args.reference_dir is not None:
        ref_paths = sorted([
            str(p) for p in Path(args.reference_dir).iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ])

    inference_size = (512, 512)

    image = Image.open(args.validation_image).convert("RGB")
    image = image.resize(inference_size, Image.LANCZOS)

    binary_mask_image = Image.open(args.validation_mask).convert("L")
    binary_mask_image = binary_mask_image.point(lambda p: 255 if p > 127 else 0)
    binary_mask_image = binary_mask_image.resize(inference_size, Image.NEAREST)

    mask_image = binary_mask_image.copy()
    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)

    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)

    candidates = []

    for idx in range(args.num_candidates):
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        else:
            generator = None

        result = pipe(
            prompt="a photo of sks",
            image=image,
            mask_image=mask_image,
            num_inference_steps=200,
            guidance_scale=1,
            generator=generator,
        ).images[0]

        if result.size != image.size:
            result = result.resize(image.size, Image.LANCZOS)

        composited = Image.composite(result, image, mask_image)

        score = None
        if len(ref_paths) > 0:
            score = correspondence_score(
                composited,
                ref_paths,
                binary_mask_image,
                args.match_threshold,
            )

        candidates.append({
            "candidate_idx": idx,
            "score": score,
            "image": composited,
        })

    if len(ref_paths) > 0:
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    kept = candidates[:args.top_k]

    for rank, item in enumerate(kept):
        item["image"].save(f"{args.output_dir}/{rank:02d}.png")

    if args.scores_json is not None:
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
    del pipe
    torch.cuda.empty_cache()

# python infer.py `
#   --model_path="bench0-model" `
#   --validation_image="realfill_dataset/RealBench/0/target/target.png" `
#   --validation_mask="realfill_dataset/RealBench/0/target/mask.png" `
#   --reference_dir="realfill_dataset/RealBench/0/ref" `
#   --output_dir="bench0-32ranked" `
#   --num_candidates=32 `
#   --top_k=32 `
#   --seed=42 `
#   --scores_json="bench0-32ranked/scores.json"




