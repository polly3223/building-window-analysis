#!/usr/bin/env python3
"""
Building Window Analysis Pipeline
==================================
Full pipeline: clean image → segment windows/facade → calculate ratio

Usage:
    uv run python analyze.py <input_image> [output_dir]

Steps:
    1. Gemini 2.5 Flash Image cleans the building photo (removes trees, fixes perspective)
    2. Gemini generates a windows mask (red overlay)
    3. Gemini generates a facade mask (blue on black)
    4. Pixel counting calculates window-to-facade area ratio
"""
import sys
from pathlib import Path

from step1_clean import clean_building
from step2_segment import gemini_segment, GEMINI_API_KEY
from step3_calculate import calculate_ratio
from PIL import Image
from google import genai


def analyze(input_path: str, output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Clean
    print("=" * 60)
    print("  STEP 1: Cleaning building image with Gemini")
    print("=" * 60)
    cleaned_path = str(out / "cleaned.png")
    clean_building(input_path, cleaned_path)

    # Step 2: Segment
    print("\n" + "=" * 60)
    print("  STEP 2: Segmenting windows and facade")
    print("=" * 60)
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(cleaned_path)

    # Windows
    print("\n[Windows mask]")
    windows_prompt = """Take this building image and create a MASK overlay:
- Paint ALL glass windows and glass doors with SOLID BRIGHT RED (#FF0000)
- Paint everything else (walls, balconies, roof, sky, ground) as SOLID BLACK (#000000)
- Be thorough - mark EVERY visible window including small ones
- The output should be a flat color mask, not a photo - just red shapes on black background
- Same dimensions as the input image"""

    windows_path = str(out / "mask_windows.png")
    gemini_segment(client, img, windows_prompt, windows_path)

    # Facade
    print("\n[Facade mask]")
    facade_prompt = """Take this building image and create a MASK overlay:
- Paint the ENTIRE building facade (walls, windows, balconies, doors - everything that is part of the building structure from ground to roof) with SOLID BRIGHT BLUE (#0000FF)
- Paint everything else (sky, ground/sidewalk, any background) as SOLID BLACK (#000000)
- The output should be a flat color mask - just blue shape on black background
- Same dimensions as the input image"""

    facade_path = str(out / "mask_facade.png")
    gemini_segment(client, img, facade_prompt, facade_path)

    # Step 3: Calculate
    print("\n" + "=" * 60)
    print("  STEP 3: Calculating window-to-facade ratio")
    print("=" * 60)
    result = calculate_ratio(cleaned_path, windows_path, facade_path, str(out))

    print(f"\nDone! All outputs in: {out}/")
    print(f"  cleaned.png      — cleaned building facade")
    print(f"  mask_windows.png  — windows detection mask")
    print(f"  mask_facade.png   — facade detection mask")
    print(f"  result.png        — visualization with ratio overlay")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python analyze.py <input_image> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "."

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    analyze(input_file, output)
