"""
Step 3: Generate window (red) + wall (blue) segmentation mask.
Works on the dimmed image from step 2 — focuses on the bright building.
Instructs Gemini to mark full window rectangles even behind balconies.
"""
import os
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable")
    sys.exit(1)

PROMPT = """The bright building in this image is the one to analyze. Ignore the darkened areas.

Create a segmentation mask for energy efficiency analysis:
- WINDOWS → SOLID RED (#FF0000)
- OPAQUE WALL (brick, concrete, plaster) → SOLID BLUE (#0000FF)
- EVERYTHING ELSE → SOLID BLACK (#000000)

ABOUT BALCONIES AND WINDOWS:
This building has balconies. Due to the camera angle from street level, the balcony floor slabs COVER the upper portion of the windows on the floor below. But the windows are still there — they are standard rectangular windows that extend upward behind the balcony slab.

For EACH window:
- Look at the visible bottom part of the window below each balcony
- The full window is a rectangle that extends upward — the top part is just hidden behind the concrete balcony slab
- Mark the ENTIRE window rectangle in red, including the area currently hidden behind the balcony
- The balcony slab itself (the horizontal concrete platform) should be BLACK, not blue — it is not part of the vertical facade wall

Think of it this way: if you removed all the balconies, each window would be a tall rectangle. Mark that full tall rectangle in red.

Output: flat color mask — solid red, blue, black only. Same dimensions as input."""


def mask(input_path: str, output_path: str) -> Image.Image:
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(input_path)
    print(f"  Input: {img.size[0]}x{img.size[1]}")

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[PROMPT, img],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"  Gemini: {part.text[:150]}")
        elif part.inline_data is not None:
            result = Image.open(BytesIO(part.inline_data.data))
            result.save(output_path)
            print(f"  Saved: {output_path} ({result.size[0]}x{result.size[1]})")
            return result

    print("  ERROR: No image returned")
    sys.exit(1)


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "step2_output.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "step3_mask.png"
    if not Path(inp).exists():
        print(f"Error: {inp} not found")
        sys.exit(1)
    print("[Step 3] Generating mask...")
    mask(inp, out)
