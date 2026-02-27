"""
Step 2: Use Gemini to generate segmentation masks.
Pass 1: Generate image with windows highlighted in bright red
Pass 2: Generate image with full building facade highlighted in bright blue
Then count pixels to calculate ratios.
"""
import os
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED")


def gemini_segment(client, image: Image.Image, prompt: str, output_path: str) -> Image.Image:
    """Ask Gemini to produce a segmentation overlay."""
    print(f"  Sending segmentation request...")
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"  Gemini: {part.text[:200]}")
        elif part.inline_data is not None:
            result = Image.open(BytesIO(part.inline_data.data))
            result.save(output_path)
            print(f"  Saved: {output_path} ({result.size[0]}x{result.size[1]})")
            return result

    print("  ERROR: No image returned")
    sys.exit(1)


def count_color_pixels(img: Image.Image, target_rgb: tuple, tolerance: int = 60) -> int:
    """Count pixels close to a target color."""
    arr = np.array(img.convert("RGB"))
    diff = np.abs(arr.astype(int) - np.array(target_rgb).astype(int))
    mask = np.all(diff < tolerance, axis=2)
    return int(np.sum(mask))


def segment_and_calculate(cleaned_path: str, output_dir: str = "."):
    """Run segmentation on the cleaned building image."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(cleaned_path)

    # Pass 1: Windows mask
    print("\n[Pass 1] Segmenting WINDOWS...")
    windows_prompt = """Take this building image and create a MASK overlay:
- Paint ALL glass windows and glass doors with SOLID BRIGHT RED (#FF0000)
- Paint everything else (walls, balconies, roof, sky, ground) as SOLID BLACK (#000000)
- Be thorough - mark EVERY visible window including small ones
- The output should be a flat color mask, not a photo - just red shapes on black background
- Same dimensions as the input image"""

    windows_img = gemini_segment(
        client, img, windows_prompt,
        os.path.join(output_dir, "mask_windows.png")
    )

    # Pass 2: Building facade mask
    print("\n[Pass 2] Segmenting BUILDING FACADE...")
    facade_prompt = """Take this building image and create a MASK overlay:
- Paint the ENTIRE building facade (walls, windows, balconies, doors - everything that is part of the building structure from ground to roof) with SOLID BRIGHT BLUE (#0000FF)
- Paint everything else (sky, ground/sidewalk, any background) as SOLID BLACK (#000000)
- The output should be a flat color mask - just blue shape on black background
- Same dimensions as the input image"""

    facade_img = gemini_segment(
        client, img, facade_prompt,
        os.path.join(output_dir, "mask_facade.png")
    )

    # Count pixels
    print("\n[Calculating areas]")

    # Windows: count red pixels
    window_pixels = count_color_pixels(windows_img, (255, 0, 0), tolerance=80)

    # Facade: count blue pixels
    facade_pixels = count_color_pixels(facade_img, (0, 0, 255), tolerance=80)

    total_pixels = img.size[0] * img.size[1]

    print(f"  Total image pixels: {total_pixels:,}")
    print(f"  Window pixels (red): {window_pixels:,}")
    print(f"  Facade pixels (blue): {facade_pixels:,}")

    if facade_pixels > 0:
        ratio = window_pixels / facade_pixels
        print(f"\n  ═══════════════════════════════════════")
        print(f"  WINDOW-TO-FACADE RATIO: {ratio:.1%}")
        print(f"  ═══════════════════════════════════════")
        print(f"  Window area: {ratio:.1%} of total facade")
        print(f"  Wall area:   {1 - ratio:.1%} of total facade")
    else:
        print("  ERROR: No facade pixels detected")

    # Also create a combined visualization
    print("\n[Creating visualization]")
    vis = np.array(img.convert("RGB")).copy()

    # Overlay windows in semi-transparent red
    w_arr = np.array(windows_img.convert("RGB"))
    w_mask = np.all(np.abs(w_arr.astype(int) - [255, 0, 0]) < 80, axis=2)
    vis[w_mask] = (vis[w_mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

    # Overlay facade outline in blue
    f_arr = np.array(facade_img.convert("RGB"))
    f_mask = np.all(np.abs(f_arr.astype(int) - [0, 0, 255]) < 80, axis=2)
    # Only color facade edge, not interior (so windows show clearly)
    facade_only = f_mask & ~w_mask
    vis[facade_only] = (vis[facade_only] * 0.7 + np.array([0, 100, 255]) * 0.3).astype(np.uint8)

    vis_img = Image.fromarray(vis)
    vis_path = os.path.join(output_dir, "visualization.png")
    vis_img.save(vis_path)
    print(f"  Visualization saved: {vis_path}")

    return {
        "window_pixels": window_pixels,
        "facade_pixels": facade_pixels,
        "ratio": ratio if facade_pixels > 0 else 0,
    }


if __name__ == "__main__":
    cleaned = sys.argv[1] if len(sys.argv) > 1 else "cleaned.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not Path(cleaned).exists():
        print(f"Error: {cleaned} not found. Run step1_clean.py first.")
        sys.exit(1)

    segment_and_calculate(cleaned, out_dir)
