"""
Step 3: Calculate window-to-facade area ratio from the masks.
Uses the difference between original and windows-mask to detect red overlay,
and uses blue-channel dominance for facade detection.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calculate_ratio(cleaned_path: str, windows_mask_path: str, facade_mask_path: str, output_dir: str = "."):
    """Calculate window-to-facade ratio from masks."""

    orig = np.array(Image.open(cleaned_path).convert("RGB")).astype(float)
    windows_img = np.array(Image.open(windows_mask_path).convert("RGB")).astype(float)
    facade_img = np.array(Image.open(facade_mask_path).convert("RGB")).astype(int)

    # ── Detect window pixels ────────────────────────────────
    # Gemini overlaid red on the photo — find pixels where red channel increased
    red_diff = windows_img[:, :, 0] - orig[:, :, 0]
    window_mask = red_diff > 30  # Red channel increased by 30+
    window_pixels = int(np.sum(window_mask))

    # ── Detect facade pixels ────────────────────────────────
    # Gemini made a blue mask — find blue-dominant non-black pixels
    facade_mask = (
        (facade_img[:, :, 2] > 80) &       # Blue channel significant
        (facade_img[:, :, 2] > facade_img[:, :, 0] + 20) &  # Blue > Red
        (facade_img[:, :, 2] > facade_img[:, :, 1])          # Blue > Green
    )
    facade_pixels = int(np.sum(facade_mask))

    total_pixels = orig.shape[0] * orig.shape[1]

    # ── Print results ───────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  BUILDING ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"  Image size:      {orig.shape[1]} x {orig.shape[0]}")
    print(f"  Total pixels:    {total_pixels:>12,}")
    print(f"  Facade pixels:   {facade_pixels:>12,} ({facade_pixels/total_pixels:.1%} of image)")
    print(f"  Window pixels:   {window_pixels:>12,} ({window_pixels/total_pixels:.1%} of image)")
    print(f"{'='*50}")

    if facade_pixels > 0:
        ratio = window_pixels / facade_pixels
        wall_pixels = facade_pixels - window_pixels
        print(f"  WINDOW / FACADE: {ratio:.1%}")
        print(f"  WALL / FACADE:   {1-ratio:.1%}")
        print(f"{'='*50}")
        print(f"  Window area ≈ {ratio*100:.1f}% of facade")
        print(f"  Opaque wall  ≈ {(1-ratio)*100:.1f}% of facade")
    else:
        ratio = 0
        print("  ERROR: Could not detect facade")

    # ── Create visualization ────────────────────────────────
    vis = np.array(Image.open(cleaned_path).convert("RGB")).copy()

    # Blue tint for facade
    vis[facade_mask] = (vis[facade_mask] * 0.7 + np.array([50, 100, 255]) * 0.3).astype(np.uint8)
    # Red overlay for windows
    vis[window_mask] = (vis[window_mask] * 0.3 + np.array([255, 30, 30]) * 0.7).astype(np.uint8)

    vis_pil = Image.fromarray(vis)

    # Add text label
    draw = ImageDraw.Draw(vis_pil)
    text = f"Windows: {ratio:.1%} of facade"
    draw.rectangle([10, 10, 400, 50], fill=(0, 0, 0, 180))
    draw.text((20, 18), text, fill=(255, 255, 255))

    vis_path = f"{output_dir}/result.png"
    vis_pil.save(vis_path)
    print(f"\n  Visualization: {vis_path}")

    return {"window_pixels": window_pixels, "facade_pixels": facade_pixels, "ratio": ratio}


if __name__ == "__main__":
    cleaned = sys.argv[1] if len(sys.argv) > 1 else "cleaned.png"
    windows = sys.argv[2] if len(sys.argv) > 2 else "mask_windows.png"
    facade = sys.argv[3] if len(sys.argv) > 3 else "mask_facade.png"
    out = sys.argv[4] if len(sys.argv) > 4 else "."

    calculate_ratio(cleaned, windows, facade, out)
