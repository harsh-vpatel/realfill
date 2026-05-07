import json
import glob
import os
from statistics import mean

# Metrics to aggregate
METRICS = ["PSNR", "SSIM", "LPIPS", "DreamSim", "DINO", "CLIP"]

# Pattern for your evaluation files
PATTERN = "bench*_ramr2_final_eval.json"

# Which image names to aggregate across all files
TARGET_IMAGES = ["00.png", "01.png"]


def main():
    files = sorted(glob.glob(PATTERN))

    if not files:
        print(f"No files found matching pattern: {PATTERN}")
        return

    # Store per-file extracted rows
    per_file_rows = []

    # Buckets for averaging across files, grouped by image name
    grouped_metrics = {
        image_name: {m: [] for m in METRICS}
        for image_name in TARGET_IMAGES
    }

    # Track how many files contributed to each image name
    grouped_counts = {image_name: 0 for image_name in TARGET_IMAGES}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            per_image = data.get("per_image", None)
            if per_image is None or not isinstance(per_image, list):
                print(f"Skipping {path}: no valid 'per_image' field found")
                continue

            # Build a quick lookup by image filename
            per_image_lookup = {}
            for item in per_image:
                img_name = item.get("image", None)
                if img_name is not None:
                    per_image_lookup[img_name] = item

            row = {"file": os.path.basename(path)}
            has_any_target = False

            for image_name in TARGET_IMAGES:
                if image_name not in per_image_lookup:
                    # It's okay if a given file doesn't contain this image
                    continue

                item = per_image_lookup[image_name]
                has_any_target = True

                row[image_name] = {}
                missing_metric = False

                for m in METRICS:
                    if m not in item:
                        print(f"Skipping metric '{m}' for {image_name} in {path}")
                        missing_metric = True
                        break

                    value = float(item[m])
                    row[image_name][m] = value
                    grouped_metrics[image_name][m].append(value)

                if not missing_metric:
                    grouped_counts[image_name] += 1

            if has_any_target:
                per_file_rows.append(row)

        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not per_file_rows:
        print("No valid evaluation JSON files were loaded.")
        return

    # Compute averages for each target image
    grouped_averages = {}
    for image_name in TARGET_IMAGES:
        if grouped_counts[image_name] == 0:
            grouped_averages[image_name] = None
        else:
            grouped_averages[image_name] = {
                m: mean(grouped_metrics[image_name][m])
                for m in METRICS
            }

    # ------------------------------------------------------------------
    # Print per-file extracted metrics
    # ------------------------------------------------------------------
    print("=" * 120)
    print("Per-file selected image metrics")
    print("=" * 120)
    for row in per_file_rows:
        print(f"\n{row['file']}")
        for image_name in TARGET_IMAGES:
            if image_name in row:
                vals = row[image_name]
                print(
                    f"  {image_name:6s} | "
                    f"PSNR={vals['PSNR']:.6f} | "
                    f"SSIM={vals['SSIM']:.6f} | "
                    f"LPIPS={vals['LPIPS']:.6f} | "
                    f"DreamSim={vals['DreamSim']:.6f} | "
                    f"DINO={vals['DINO']:.6f} | "
                    f"CLIP={vals['CLIP']:.6f}"
                )

    # ------------------------------------------------------------------
    # Print grouped averages
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("Average across all files by image name")
    print("=" * 120)

    for image_name in TARGET_IMAGES:
        print(f"\nImage: {image_name}")
        if grouped_averages[image_name] is None:
            print("  No data found.")
            continue

        print(f"  Num files contributing: {grouped_counts[image_name]}")
        for m in METRICS:
            print(f"  {m:10s}: {grouped_averages[image_name][m]:.6f}")

    # ------------------------------------------------------------------
    # Save aggregated result
    # ------------------------------------------------------------------
    output = {
        "pattern": PATTERN,
        "target_images": TARGET_IMAGES,
        "num_files_scanned": len(files),
        "num_valid_files": len(per_file_rows),
        "per_file_rows": per_file_rows,
        "grouped_counts": grouped_counts,
        "grouped_averages": grouped_averages,
    }

    out_path = "all_bench_ramr2_final_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved aggregate JSON to: {out_path}")


if __name__ == "__main__":
    main()