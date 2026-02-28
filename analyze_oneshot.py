"""
One-shot building window analysis for energy efficiency.
Two Gemini calls in a single script:
  1. Clean the Google Maps photo (remove trees/cars, reveal facade)
  2. Generate combined window (red) + wall (blue) mask — balconies excluded
Then pixel-count for the ratio.
"""
import os
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable")
    sys.exit(1)


def gemini_image(client, prompt: str, image: Image.Image, label: str) -> Image.Image:
    """Send image + prompt to Gemini, return the generated image."""
    print(f"  [{label}] Sending to Gemini...")
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"  [{label}] Gemini: {part.text[:200]}")
        elif part.inline_data is not None:
            result = Image.open(BytesIO(part.inline_data.data))
            print(f"  [{label}] Got image: {result.size[0]}x{result.size[1]}")
            return result
    print(f"  [{label}] ERROR: No image returned")
    sys.exit(1)


def analyze_building(input_path: str, output_dir: str = ".") -> dict:
    """Clean building photo, generate mask, calculate window ratio."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    input_image = Image.open(input_path)
    w, h = input_image.size
    print(f"Input: {w}x{h}")

    # ── Step 1: Clean the photo ──────────────────────────────
    print("\n[Step 1] Cleaning photo...")
    clean_prompt = """Remove the trees and cars from this photo. Replace them with the building facade that is behind them.

CRITICAL CONSTRAINTS:
- The building must stay the EXACT SAME SIZE and shape as in the original photo
- Do NOT make the building wider or taller — it should occupy the same area of the image
- Do NOT change the perspective or camera angle
- Keep the same framing and composition — same sky, same ground area
- Where a tree was, show the building wall/windows that were hidden behind it, matching the style of the visible parts
- The black triangular areas are panoramic stitch artifacts — replace them with sky
- Do NOT crop, do NOT zoom in, do NOT reframe
- The output image should be the same scene, same size, just without the trees and cars"""

    cleaned = gemini_image(client, clean_prompt, input_image, "CLEAN")
    cleaned_path = os.path.join(output_dir, "cleaned_oneshot.png")
    cleaned.save(cleaned_path)

    # ── Step 2: Generate combined mask ───────────────────────
    print("\n[Step 2] Generating segmentation mask...")
    mask_prompt = """There are TWO buildings in this image separated by a gap/sky. I need you to analyze ONLY the LEFT building (the red/brown brick one). The RIGHT building must be completely IGNORED — paint it black.

Create a segmentation mask for energy efficiency analysis of ONLY the LEFT building:
- LEFT building WINDOWS (glass surfaces only) → SOLID RED (#FF0000)
- LEFT building OPAQUE WALL (flat vertical facade surface: brick, concrete, plaster) → SOLID BLUE (#0000FF)
- EVERYTHING ELSE → SOLID BLACK (#000000)

What must be BLACK:
- The entire RIGHT building — all of it, walls and windows
- Sky, ground, sidewalk, street
- Balcony slabs (horizontal protruding concrete platforms)
- Balcony railings, fences, awnings
- Roof, chimneys, antennas
- Shopfronts / ground floor commercial (glass storefronts at street level)

Only the FLAT VERTICAL FACADE of the LEFT building counts.
Output: flat color mask, solid colors on pure black, same dimensions as input."""

    mask = gemini_image(client, mask_prompt, cleaned, "MASK")
    mask_path = os.path.join(output_dir, "mask_oneshot.png")
    mask.save(mask_path)

    if mask is None:
        print("ERROR: Gemini did not return an image")
        sys.exit(1)

    # ── Pixel counting ───────────────────────────────────────
    arr = np.array(mask.convert("RGB")).astype(int)

    # Windows: red-dominant pixels
    window_mask = (
        (arr[:, :, 0] > 80) &
        (arr[:, :, 0] > arr[:, :, 1] + 30) &
        (arr[:, :, 0] > arr[:, :, 2] + 30)
    )
    window_px = int(np.sum(window_mask))

    # Facade wall: blue-dominant pixels (various blue/cyan shades)
    wall_mask = (
        (arr[:, :, 2] > 60) &
        (arr[:, :, 2] > arr[:, :, 0] + 15) &
        ((arr[:, :, 2] > arr[:, :, 1]) | (arr[:, :, 1] > 60))
    )
    wall_px = int(np.sum(wall_mask))

    facade_px = window_px + wall_px
    total_px = arr.shape[0] * arr.shape[1]

    # ── Results ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  ENERGY EFFICIENCY — WINDOW ANALYSIS")
    print(f"{'='*50}")
    print(f"  Image:           {arr.shape[1]} x {arr.shape[0]}")
    print(f"  Facade total:    {facade_px:>10,} px ({facade_px/total_px:.1%} of image)")
    print(f"  ├─ Windows:      {window_px:>10,} px")
    print(f"  └─ Opaque wall:  {wall_px:>10,} px")
    print(f"{'='*50}")

    if facade_px > 0:
        w_ratio = window_px / facade_px
        print(f"  WINDOW / FACADE:  {w_ratio:.1%}")
        print(f"  WALL / FACADE:    {1 - w_ratio:.1%}")
        print(f"{'='*50}")
    else:
        w_ratio = 0
        print("  ERROR: No facade detected")

    # ── Visualization ────────────────────────────────────────
    # Overlay on the cleaned image (not original — trees would hide the overlay)
    orig = np.array(cleaned.convert("RGB")).copy()
    # Resize mask to match original if needed
    if mask.size != (orig.shape[1], orig.shape[0]):
        mask_resized = mask.resize((orig.shape[1], orig.shape[0]), Image.NEAREST)
        arr_r = np.array(mask_resized.convert("RGB")).astype(int)
        window_mask_r = (
            (arr_r[:, :, 0] > 80) &
            (arr_r[:, :, 0] > arr_r[:, :, 1] + 30) &
            (arr_r[:, :, 0] > arr_r[:, :, 2] + 30)
        )
        wall_mask_r = (
            (arr_r[:, :, 2] > 60) &
            (arr_r[:, :, 2] > arr_r[:, :, 0] + 15) &
            ((arr_r[:, :, 2] > arr_r[:, :, 1]) | (arr_r[:, :, 1] > 60))
        )
    else:
        window_mask_r = window_mask
        wall_mask_r = wall_mask

    # Red tint on windows
    orig[window_mask_r] = (orig[window_mask_r] * 0.3 + np.array([255, 30, 30]) * 0.7).astype(np.uint8)
    # Blue tint on wall
    orig[wall_mask_r] = (orig[wall_mask_r] * 0.5 + np.array([50, 100, 255]) * 0.5).astype(np.uint8)

    vis = Image.fromarray(orig)
    draw = ImageDraw.Draw(vis)

    # Label
    label = f"Windows: {w_ratio:.1%} | Wall: {1-w_ratio:.1%}  (balconies excluded)"
    draw.rectangle([5, 5, 520, 35], fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))

    result_path = os.path.join(output_dir, "result_oneshot.png")
    vis.save(result_path)
    print(f"\n  Visualization: {result_path}")

    return {
        "window_pixels": window_px,
        "wall_pixels": wall_px,
        "facade_pixels": facade_px,
        "window_ratio": w_ratio,
    }


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    analyze_building(input_file, out_dir)
