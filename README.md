# Building Window Analysis

Calculate the window-to-facade area ratio of buildings from street-level photos — even with bad perspective, trees, and obstructions.

## How it works

1. **Gemini 2.5 Flash Image** cleans the photo: removes trees, cars, fixes perspective distortion → clean front-facing facade
2. **Gemini** generates a **windows mask** (red overlay on detected windows)
3. **Gemini** generates a **facade mask** (blue mask of the full building)
4. **Pixel counting** calculates the window/facade area ratio

## Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"
```

## Usage

```bash
# Full pipeline (clean → segment → calculate)
uv run python analyze.py photo.jpg output/

# Or run steps individually:
uv run python step1_clean.py photo.jpg cleaned.png
uv run python step2_segment.py cleaned.png output/
uv run python step3_calculate.py cleaned.png mask_windows.png mask_facade.png output/
```

## Output

- `cleaned.png` — cleaned building facade (perspective-corrected, obstructions removed)
- `mask_windows.png` — windows detection mask
- `mask_facade.png` — facade detection mask
- `result.png` — visualization with ratio overlay

## Example Result

From a Google Street View photo with trees blocking the building:

| Step | Output |
|------|--------|
| Input | Bad perspective, trees covering facade |
| Cleaned | Clean front-facing facade |
| Result | **Windows ≈ 15% of facade, Wall ≈ 85%** |

## Requirements

- Python 3.12+
- [Google Gemini API key](https://ai.google.dev/) (uses `gemini-2.5-flash-image` model)
- ~$0.12 per image (3 Gemini API calls)

## Tech Stack

- **Gemini 2.5 Flash Image** (Nano Banana) — image editing + mask generation
- **Pillow** — image processing
- **NumPy** — pixel counting
- **uv** — Python package management
