"""
Step 1: Use Gemini Flash Image to clean building photos.
Removes trees, cars, and obstructions to reveal the existing facade.
Does NOT invent or reconstruct — only removes what's in front of the building.
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

def clean_building(input_path: str, output_path: str) -> str:
    """Send image to Gemini with a cleanup prompt, save the result."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    input_image = Image.open(input_path)
    print(f"Input image: {input_image.size[0]}x{input_image.size[1]}")

    prompt = """Remove the trees and cars from this photo. Replace them with the building facade that is behind them.

CRITICAL CONSTRAINTS:
- The building must stay the EXACT SAME SIZE and shape as in the original photo
- Do NOT make the building wider or taller — it should occupy the same area of the image
- Do NOT change the perspective or camera angle
- Keep the same framing and composition — same sky, same ground area
- Where a tree was, show the building wall/windows that were hidden behind it, matching the style of the visible parts
- The black triangular areas are panoramic stitch artifacts — replace them with sky
- Do NOT crop, do NOT zoom in, do NOT reframe
- The output image should be the same scene, same size, just without the trees and cars"""

    print("Sending to Gemini Flash Image...")
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt, input_image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    # Extract result
    saved = False
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(f"Gemini says: {part.text}")
        elif part.inline_data is not None:
            edited = Image.open(BytesIO(part.inline_data.data))
            edited.save(output_path)
            print(f"Cleaned image saved: {output_path} ({edited.size[0]}x{edited.size[1]})")
            saved = True

    if not saved:
        print("ERROR: Gemini did not return an image. Response:")
        print(response)
        sys.exit(1)

    return output_path


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cleaned.png"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    clean_building(input_file, output_file)
