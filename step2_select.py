"""
Step 2: Isolate the center building by darkening everything else.
The center building stays at full brightness, untouched.
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

PROMPT = """Darken everything in this image EXCEPT the narrow building in the CENTER (the red/brown brick one between the two other buildings).

- Make the left building very dark/dimmed
- Make the right building very dark/dimmed
- Make the sky very dark
- Make the ground very dark

The center building should remain at full brightness, completely untouched — keep all its details, balconies, windows exactly as they are. Just dim everything around it so it stands out."""


def select(input_path: str, output_path: str) -> Image.Image:
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
    inp = sys.argv[1] if len(sys.argv) > 1 else "step1_output.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "step2_output.png"
    if not Path(inp).exists():
        print(f"Error: {inp} not found")
        sys.exit(1)
    print("[Step 2] Selecting center building...")
    select(inp, out)
