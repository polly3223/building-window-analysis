"""
Full pipeline: input image → window/wall ratio.
Calls step1, step2, step3 in sequence, then does pixel math.

Usage:
    GEMINI_API_KEY=... uv run run_pipeline.py input.jpg [output_dir]
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from step1_clean import clean
from step2_select import select
from step3_mask import mask


def count_pixels(mask_path: str) -> dict:
    """Count red (window) and blue (wall) pixels in the mask."""
    arr = np.array(Image.open(mask_path).convert("RGB")).astype(int)

    window_mask = (
        (arr[:, :, 0] > 80) &
        (arr[:, :, 0] > arr[:, :, 1] + 30) &
        (arr[:, :, 0] > arr[:, :, 2] + 30)
    )
    window_px = int(np.sum(window_mask))

    wall_mask = (
        (arr[:, :, 2] > 60) &
        (arr[:, :, 2] > arr[:, :, 0] + 15) &
        ((arr[:, :, 2] > arr[:, :, 1]) | (arr[:, :, 1] > 60))
    )
    wall_px = int(np.sum(wall_mask))

    return {
        "window_px": window_px,
        "wall_px": wall_px,
        "window_mask": window_mask,
        "wall_mask": wall_mask,
    }


def make_visualization(base_path: str, mask_path: str, output_path: str, w_ratio: float):
    """Overlay mask on the step2 image for visualization."""
    orig = np.array(Image.open(base_path).convert("RGB")).copy()
    mask_img = Image.open(mask_path).convert("RGB")

    # Resize mask if needed
    if mask_img.size != (orig.shape[1], orig.shape[0]):
        mask_img = mask_img.resize((orig.shape[1], orig.shape[0]), Image.NEAREST)

    arr = np.array(mask_img).astype(int)
    wm = (arr[:, :, 0] > 80) & (arr[:, :, 0] > arr[:, :, 1] + 30) & (arr[:, :, 0] > arr[:, :, 2] + 30)
    bm = (arr[:, :, 2] > 60) & (arr[:, :, 2] > arr[:, :, 0] + 15)

    orig[wm] = (orig[wm] * 0.3 + np.array([255, 30, 30]) * 0.7).astype(np.uint8)
    orig[bm] = (orig[bm] * 0.5 + np.array([50, 100, 255]) * 0.5).astype(np.uint8)

    vis = Image.fromarray(orig)
    draw = ImageDraw.Draw(vis)
    label = f"Windows: {w_ratio:.1%} | Wall: {1-w_ratio:.1%}"
    draw.rectangle([5, 5, 350, 35], fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))
    vis.save(output_path)


def run(input_path: str, output_dir: str = "."):
    """Run the full 3-step pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    step1_out = os.path.join(output_dir, "step1_output.png")
    step2_out = os.path.join(output_dir, "step2_output.png")
    step3_out = os.path.join(output_dir, "step3_mask.png")
    result_out = os.path.join(output_dir, "result.png")

    # Step 1: Remove obstructions
    print("\n" + "="*50)
    print("[Step 1] Removing obstructions...")
    print("="*50)
    clean(input_path, step1_out)

    # Step 2: Isolate center building
    print("\n" + "="*50)
    print("[Step 2] Selecting center building...")
    print("="*50)
    select(step1_out, step2_out)

    # Step 3: Generate mask
    print("\n" + "="*50)
    print("[Step 3] Generating mask...")
    print("="*50)
    mask(step2_out, step3_out)

    # Pixel math
    print("\n" + "="*50)
    print("[Results] Counting pixels...")
    print("="*50)
    pixels = count_pixels(step3_out)
    window_px = pixels["window_px"]
    wall_px = pixels["wall_px"]
    facade_px = window_px + wall_px

    mask_img = Image.open(step3_out)
    total_px = mask_img.size[0] * mask_img.size[1]

    print(f"\n  Image:         {mask_img.size[0]} x {mask_img.size[1]}")
    print(f"  Facade total:  {facade_px:>10,} px ({facade_px/total_px:.1%} of image)")
    print(f"  ├─ Windows:    {window_px:>10,} px")
    print(f"  └─ Wall:       {wall_px:>10,} px")

    if facade_px > 0:
        w_ratio = window_px / facade_px
        print(f"\n  ╔══════════════════════════════╗")
        print(f"  ║  WINDOW / FACADE:  {w_ratio:>6.1%}   ║")
        print(f"  ║  WALL / FACADE:    {1-w_ratio:>6.1%}   ║")
        print(f"  ╚══════════════════════════════╝")
    else:
        w_ratio = 0
        print("\n  ERROR: No facade detected")

    # Visualization
    make_visualization(step2_out, step3_out, result_out, w_ratio)
    print(f"\n  Files:")
    print(f"    Step 1 (cleaned):  {step1_out}")
    print(f"    Step 2 (selected): {step2_out}")
    print(f"    Step 3 (mask):     {step3_out}")
    print(f"    Result:            {result_out}")

    return w_ratio


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    run(input_file, out_dir)
