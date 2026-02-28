"""
One-shot building window analysis for energy efficiency.
Single Gemini call: raw Google Maps photo → segmentation mask.
Then pixel-count for window/wall ratio.
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


def analyze_building(input_path: str, output_dir: str = ".") -> dict:
    """Single Gemini call → mask → pixel math."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    input_image = Image.open(input_path)
    w, h = input_image.size
    print(f"Input: {w}x{h}")

    # ── Single Gemini call: generate mask ─────────────────────
    print("\nGenerating segmentation mask...")
    prompt = """This is a Google Maps street view photo. There are buildings visible, possibly partially hidden by trees, cars, or other obstructions. There is a building facade in the CENTER of the image — this is the one I want analyzed.

WHICH BUILDING: ONLY the facade visible in the center of the photo. If there are buildings on the left and right edges, IGNORE them completely — paint them black. Only analyze the center building that the camera is pointing at.

Create a segmentation mask for energy efficiency analysis:
- That center building's WINDOWS → SOLID RED (#FF0000)
- That center building's OPAQUE WALL (vertical facade surface: brick, concrete, plaster) → SOLID BLUE (#0000FF)
- EVERYTHING ELSE → SOLID BLACK (#000000)

CRITICAL — LOOK THROUGH OBSTRUCTIONS:
- Trees, cars, and street furniture may be blocking parts of the building
- You must ESTIMATE the full facade behind these obstructions
- If you can see windows on visible parts of the building, infer that the pattern continues behind the trees
- Mark the estimated windows and wall behind obstructions as if the obstructions were transparent

CRITICAL — FULL WINDOW SIZE BEHIND BALCONIES:
- Balcony slabs from upper floors often hide the top part of windows on the floor below (due to perspective from street level)
- You MUST mark each window as a FULL RECTANGLE — include the part hidden behind the balcony above
- Imagine the balconies are transparent: draw the complete window shape
- The balcony slabs and railings themselves are BLACK (not wall, not window)

ALSO BLACK (not part of facade):
- Buildings on the left and right edges of the photo
- Sky, ground, sidewalk, street
- Trees, cars, people, street furniture
- Balcony slabs (horizontal protruding concrete) and railings
- Black triangular panoramic stitch artifacts

Output: flat color mask — solid red, blue, black only. No photo, no textures. Same dimensions as input."""

    print("  Sending to Gemini...")
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt, input_image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    mask = None
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"  Gemini: {part.text[:200]}")
        elif part.inline_data is not None:
            mask = Image.open(BytesIO(part.inline_data.data))
            mask_path = os.path.join(output_dir, "mask_oneshot.png")
            mask.save(mask_path)
            print(f"  Mask saved: {mask_path} ({mask.size[0]}x{mask.size[1]})")

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

    # Facade wall: blue-dominant pixels (various blue/cyan shades from Gemini)
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
    orig = np.array(input_image.convert("RGB")).copy()

    # Resize mask to match original if Gemini changed dimensions
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

    label = f"Windows: {w_ratio:.1%} | Wall: {1-w_ratio:.1%}"
    draw.rectangle([5, 5, 350, 35], fill=(0, 0, 0))
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
