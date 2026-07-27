# =============================================================================
# LAB 22 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): Azure OpenAI IMAGE GENERATION & EDITING — Domain 3 of the AI-103
# exam blueprint ("Implement computer vision solutions"), specifically the
# "image generation" and "video generation" module. This chapter's
# `Labs/` folder previously had zero vision coverage even though
# `05. Section Code/` already implements this for real - this lab adapts
# that proven pattern (generate -> edit -> masked edit/inpaint) into one
# runnable script, using this folder's auth conventions (bearer token via
# `az login`, not a hardcoded key like the Section Code original).
#
# Needs a source image (`input_image.png`) in this same folder for the edit
# steps — not included; supply any PNG you like (a plain photo or graphic).
#
# Prerequisites: an Azure OpenAI image-generation deployment (e.g.
# "gpt-image-2") + `az login` + .env with AZURE_OPENAI_ENDPOINT and
# IMAGE_DEPLOYMENT_NAME.
# =============================================================================

import os                       # Env vars, clear-screen command
import base64                   # Decodes the generated image's base64 payload back into raw bytes
from pathlib import Path        # Cross-platform file path handling
from dotenv import load_dotenv   # Loads .env file variables into the environment

from openai import OpenAI                                                    # The OpenAI-compatible client, pointed at Azure via base_url (same pattern as Lab 3)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Entra ID auth helpers


def save_generated_image(response, output_path):
    """Every images.* call in this file returns the SAME shape: a base64
    string under response.data[0].b64_json. Decoding + writing it to disk
    is identical regardless of whether the image was generated, edited, or
    inpainted - so this one helper is reused by all three steps below.
    """
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    Path(output_path).write_bytes(image_bytes)
    print(f"  Saved: {output_path}")


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        load_dotenv()
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        image_deployment = os.getenv("IMAGE_DEPLOYMENT_NAME", "gpt-image-2")

        # Same auth pattern as every other lab: a bearer token derived from
        # your `az login` session, no API key stored anywhere.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        client = OpenAI(base_url=azure_openai_endpoint, api_key=token_provider)

        # --- Step 1: generate a brand-new image from a text prompt ---
        # --- Step 1: generate a brand-new image from a text prompt ---
        print("Step 1/3: Generating an image from a text prompt...")
        generate_response = client.images.generate(
            model=image_deployment,
            prompt="A professional product photo of a blue ceramic coffee mug on a wooden table, soft morning light.",
            n=1,                        # how many images to generate
            size="1024x1024",
            quality="medium",           # trades off quality vs. cost/speed - "low"/"medium"/"high"
            output_format="png",
        )
        save_generated_image(generate_response, "generated_image.png")

        # --- Step 2: edit an EXISTING image with a text instruction ---
        # --- Step 2: edit an existing image with a text instruction ---
        #
        # Requires a source image on disk (input_image.png) - this is NOT
        # the image from step 1, it's meant to be your own supplied photo.
        input_image_path = Path(__file__).parent / "input_image.png"
        if input_image_path.exists():
            print("\nStep 2/3: Editing an existing image (no mask - the whole image may change)...")
            with input_image_path.open("rb") as image_file:
                edit_response = client.images.edit(
                    model=image_deployment,
                    image=image_file,
                    prompt="Make the lighting warmer and add a soft bokeh background.",
                    size="1024x1024",
                    n=1,
                    quality="medium",
                )
            save_generated_image(edit_response, "edited_image.png")

            # --- Step 3: masked edit / inpainting - only change ONE region ---
            # --- Step 3: masked edit / inpainting ---
            #
            # A mask image (same dimensions as the source, with transparent/
            # alpha-zero pixels marking the area to REPLACE) constrains the
            # edit to just that region instead of letting the whole image
            # change, which is what makes step 2 above unpredictable.
            mask_image_path = Path(__file__).parent / "mask.png"
            if mask_image_path.exists():
                print("\nStep 3/3: Masked edit (inpainting) - only the masked region changes...")
                with input_image_path.open("rb") as image_file, mask_image_path.open("rb") as mask_file:
                    masked_response = client.images.edit(
                        model=image_deployment,
                        image=image_file,
                        mask=mask_file,
                        prompt="Replace the background with a bright blue sky.",
                        size="1024x1024",
                        n=1,
                        quality="medium",
                    )
                save_generated_image(masked_response, "masked_edit.png")
            else:
                print(f"\nStep 3/3 skipped: no mask.png found at {mask_image_path}")
                print("  (create one with the same dimensions as input_image.png, transparent")
                print("   where you want the edit applied, to try masked inpainting)")
        else:
            print(f"\nSteps 2-3 skipped: no input_image.png found at {input_image_path}")
            print("  (supply your own PNG in this folder to try editing/inpainting)")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
