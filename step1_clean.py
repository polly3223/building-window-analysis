"""
Step 1: Use Gemini 2.5 Flash Image (Nano Banana) to clean building photos.
Removes trees, cars, and obstructions. Fixes perspective to get a clean facade view.
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

    prompt = """You are an architectural photo editor. Take this building photograph and create a clean,
front-facing view of JUST the main building facade (the left building - the multi-story residential building).

Remove ALL obstructions:
- Remove all trees and vegetation covering the building
- Remove all cars and vehicles
- Remove the street/sidewalk
- Remove the black triangular borders (this is a bad panoramic stitch)
- Remove the other buildings on the right

Reconstruct the building facade as if photographed straight-on with no perspective distortion.
Show the full facade from ground level to roof. Fill in any parts hidden by trees by inferring
the building's architectural pattern (it has a repeating grid of windows and balconies).

The output should look like a clean architectural elevation drawing / front view of just that one building.
Keep it photorealistic."""

    print("Sending to Gemini 2.5 Flash Image...")
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
