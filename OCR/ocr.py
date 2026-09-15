import os
import io
import json
import time
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

# =======================
# ADDED FOR FLUTTER LINK
# =======================
API_KEY = os.environ["GIGI_API_KEY"]
# =======================
# =======================

MODEL = "gemini-3.6-flash"

client = genai.Client(api_key=API_KEY)

# =======================
# ADDED FOR FLUTTER LINK
# =======================
async def create_template(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_width, image_height = image.size
# =======================
# =======================

    prompt = """
    Perform OCR on this document.

    The document contains both Arabic and English.

    Detect EVERY visible text region that can be read.

    For each text region return:
    1. The exact text you can read.
    2. Its bounding box.
    3. Basic visual font/style information if it can be estimated.

    IMPORTANT:
    - Do NOT translate anything.
    - Keep Arabic as Arabic.
    - Keep English as English.
    - Preserve numbers, punctuation and symbols.
    - Do not invent text.
    - Include headings, labels, paragraphs, dates, fields, checkboxes text,
      signatures labels, footnotes, page numbers, etc.
    - Each bounding box must tightly surround the corresponding text.
    - Font information is an ESTIMATE from the image.
    - If the exact font family cannot be determined, return null.
    - Estimate font size relative to the image/document.
    - Do not invent a font family.

    For each text region use this format:

    {
        "text": "detected text",
        "box": [ymin, xmin, ymax, xmax],
        "font": {
            "family": null,
            "size": 24,
            "bold": false,
            "italic": false
        }
    }

    The coordinates must be normalized from 0 to 1000.

    Coordinate meaning:
    - ymin = top
    - xmin = left
    - ymax = bottom
    - xmax = right

    Font size should be an approximate pixel size based on the original image.

    If a property cannot reasonably be estimated, use null.

    Example:

    [
        {
            "text": "NAME OF APPLICANT",
            "box": [450, 100, 480, 400],
            "font": {
                "family": null,
                "size": 24,
                "bold": true,
                "italic": false
            }
        }
    ]

    Return ONLY valid JSON.
    """

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    prompt,
                    image
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            break

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed:")
            print(e)

            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All 3 attempts failed.")
                raise

    try:
        results = json.loads(response.text)

    except json.JSONDecodeError:
        print("\nGemini did not return valid JSON.")
        print("\nRaw response:")
        print(response.text)
        raise SystemExit

    print("\n========================================")
    print("OCR RESULTS")
    print("========================================\n")

    for i, item in enumerate(results, start=1):
        print(f"[{i}] {item['text']}")

    fillable_prompt = """
    Analyze this document specifically for FILLABLE INPUT AREAS.

    Your job is NOT to perform OCR.

    Detect EVERY place where a person is expected to type, write, enter,
    or sign information.

    Examples include:
    - Blank lines after labels
    - Dotted lines
    - Underscore lines such as "____________"
    - Empty rectangular input boxes
    - Blank spaces specifically intended for information
    - Name fields
    - Date fields
    - Address fields
    - Phone number fields
    - ZIP code fields
    - City/state fields
    - Medical information fields
    - Signature fields
    - Any other area clearly intended for the user to enter information

    IMPORTANT:
    - Do NOT include the text labels themselves.
    - Do NOT include normal blank space that is not an input field.
    - Do NOT include paragraphs.
    - Do NOT include headings.
    - Do NOT include printed text.
    - Do NOT include decorative lines.
    - Do NOT include checkboxes unless they are clearly intended as a
      writable/typed input area.
    - For a line such as:

        NAME OF APPLICANT
        ______________________________

      detect ONLY the underline/input area, not "NAME OF APPLICANT".

    - For an empty rectangular field, detect the inside/input area.
    - For a signature field, detect the signature line.
    - Make each bounding box tightly surround the actual writable/input area.

    Return ONLY valid JSON.

    Use this exact format:

    [
        {
            "box": [ymin, xmin, ymax, xmax]
        }
    ]

    The coordinates must be normalized from 0 to 1000.

    Coordinate meaning:
    - ymin = top
    - xmin = left
    - ymax = bottom
    - xmax = right

    Example:

    [
        {
            "box": [500, 100, 530, 600]
        }
    ]
    """

    for attempt in range(3):
        try:
            fillable_response = client.models.generate_content(
                model=MODEL,
                contents=[
                    fillable_prompt,
                    image
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            break

        except Exception as e:
            print(f"\nFillable-area attempt {attempt + 1} failed:")
            print(e)

            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All 3 fillable-area attempts failed.")
                raise

    try:
        fillable_results = json.loads(fillable_response.text)

    except json.JSONDecodeError:
        print("\nGemini did not return valid JSON for fillable areas.")
        print("\nRaw response:")
        print(fillable_response.text)
        raise SystemExit

    print("\n========================================")
    print("FILLABLE INPUT AREAS")
    print("========================================\n")

    for i, item in enumerate(fillable_results, start=1):
        print(f"[F{i}] {item['box']}")

    print("\n========================================")
    print("MAPPING TEXT TO FILLABLE AREAS")
    print("========================================\n")

    text_regions_for_mapping = []

    for i, item in enumerate(results, start=1):
        text_regions_for_mapping.append({
            "text_id": f"text_{i:03d}",
            "text": item["text"],
            "box": item["box"]
        })

    fillable_regions_for_mapping = []

    for i, item in enumerate(fillable_results, start=1):
        fillable_regions_for_mapping.append({
            "fillable_id": f"input_{i:03d}",
            "box": item["box"]
        })

    mapping_prompt = """
    You are given OCR text regions and detected fillable input areas from
    the SAME document.

    Your task is to determine which OCR text region corresponds to which
    fillable input area.

    IMPORTANT:

    1. Do NOT assume that text_001 corresponds to input_001.
    2. Do NOT assume that the ordering is the same.
    3. Use the visual/layout relationship between the text and input area.
    4. Use the meaning of the label.
    5. A label such as:
          NAME
          ADDRESS
          PHONE NUMBER
          DATE OF BIRTH
          SIGNATURE
       should be mapped to the nearby input area intended for that value.
    6. Consider areas below, above, beside, or inside a logical form section.
    7. Use semantic understanding when geometry alone is ambiguous.
    8. Do NOT map normal paragraphs, headings, instructions, page numbers,
       or unrelated text to fillable areas.
    9. A fillable area may remain unmapped if there is no clear corresponding
       label.
    10. A text region may remain unmapped if it does not correspond to a
        fillable area.
    11. Do NOT force a mapping just because two objects are close.
    12. One fillable area should normally have at most one corresponding
        label unless the document clearly indicates otherwise.
    13. If multiple text regions together form one label, choose the most
        appropriate text region or explain the relationship in the mapping.
    14. Preserve the IDs exactly as provided.

    For every OCR text region, determine whether it maps to a fillable area.

    Return ONLY valid JSON.

    Use this exact structure:

    {
        "mappings": [
            {
                "text_id": "text_001",
                "fillable_id": "input_001",
                "relationship": "label_to_input",
                "confidence": 0.95,
                "reason": "The fillable area is directly below the NAME label."
            },
            {
                "text_id": "text_002",
                "fillable_id": null,
                "relationship": "non_fillable",
                "confidence": 1.0,
                "reason": "This is a paragraph and has no associated input area."
            }
        ]
    }

    Confidence must be a number between 0 and 1.

    Use fillable_id = null when there is no corresponding fillable area.

    Do not create IDs that were not provided.

    OCR TEXT REGIONS:
    """ + json.dumps(text_regions_for_mapping, ensure_ascii=False, indent=2) + """

    FILLABLE INPUT AREAS:
    """ + json.dumps(fillable_regions_for_mapping, ensure_ascii=False, indent=2)

    for attempt in range(3):
        try:
            mapping_response = client.models.generate_content(
                model=MODEL,
                contents=[
                    mapping_prompt,
                    image
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            break

        except Exception as e:
            print(f"\nMapping attempt {attempt + 1} failed:")
            print(e)

            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All 3 mapping attempts failed.")
                raise

    try:
        mapping_results = json.loads(mapping_response.text)

    except json.JSONDecodeError:
        print("\nGemini did not return valid JSON for mapping.")
        print("\nRaw response:")
        print(mapping_response.text)
        raise SystemExit

    mappings = mapping_results.get("mappings", [])

    for mapping in mappings:
        text_id = mapping.get("text_id")
        fillable_id = mapping.get("fillable_id")
        relationship = mapping.get("relationship")
        confidence = mapping.get("confidence")

        if fillable_id is None:
            print(
                f"{text_id} -> NONE "
                f"({relationship}, confidence={confidence})"
            )
        else:
            print(
                f"{text_id} -> {fillable_id} "
                f"({relationship}, confidence={confidence})"
            )

    final_text_regions = []

    for i, item in enumerate(results, start=1):
        final_text_regions.append({
            "id": f"text_{i:03d}",
            "text": item.get("text"),
            "box": item.get("box"),
            "font": item.get("font", {
                "family": None,
                "size": None,
                "bold": None,
                "italic": None
            })
        })

    final_fillable_areas = []

    for i, item in enumerate(fillable_results, start=1):
        final_fillable_areas.append({
            "id": f"input_{i:03d}",
            "box": item.get("box")
        })

    final_json = {
        "document": {
            "image_width": image_width,
            "image_height": image_height,
            "coordinate_system": {
                "format": "normalized_0_to_1000",
                "order": "[ymin, xmin, ymax, xmax]"
            }
        },
        "text_regions": final_text_regions,
        "fillable_areas": final_fillable_areas,
        "field_mappings": mappings
    }

    print("\n========================================")
    print("TOKEN USAGE")
    print("========================================\n")

    if response.usage_metadata:
        usage = response.usage_metadata
        print("OCR Input tokens: ", usage.prompt_token_count)
        print("OCR Output tokens:", usage.candidates_token_count)
        print("OCR Total tokens: ", usage.total_token_count)
    else:
        print("OCR token usage information was not returned.")

    if fillable_response.usage_metadata:
        usage = fillable_response.usage_metadata
        print("\nFillable Input tokens: ", usage.prompt_token_count)
        print("Fillable Output tokens:", usage.candidates_token_count)
        print("Fillable Total tokens: ", usage.total_token_count)
    else:
        print("\nFillable token usage information was not returned.")

    if mapping_response.usage_metadata:
        usage = mapping_response.usage_metadata
        print("\nMapping Input tokens: ", usage.prompt_token_count)
        print("Mapping Output tokens:", usage.candidates_token_count)
        print("Mapping Total tokens: ", usage.total_token_count)
    else:
        print("\nMapping token usage information was not returned.")

    print(f"\nDetected text regions: {len(results)}")
    print(f"Detected fillable areas: {len(fillable_results)}")
    print(f"Detected mappings: {len(mappings)}")

    # =======================
    # ADDED FOR FLUTTER LINK
    # =======================
    questions = normalize_to_questions(final_json)
    return JSONResponse(questions)
    # =======================
    # =======================


# =======================
# ADDED FOR FLUTTER LINK
# =======================
def normalize_to_questions(raw: dict) -> dict:
    text_by_id = {t["id"]: t["text"] for t in raw.get("text_regions", [])}
    fillable_by_id = {f["id"]: f["box"] for f in raw.get("fillable_areas", [])}

    label_for_fillable = {}
    for mapping in raw.get("field_mappings", []):
        if mapping.get("relationship") == "label_to_input" and mapping.get("fillable_id"):
            label_text = text_by_id.get(mapping["text_id"], "")
            label_for_fillable[mapping["fillable_id"]] = label_text.replace("\n", " ").strip()

    questions = []
    for fillable_id, box in fillable_by_id.items():
        ymin, xmin, ymax, xmax = box
        questions.append({
            "region_id": fillable_id,
            "field_label": label_for_fillable.get(fillable_id, fillable_id),
            "y1": ymin,
            "x1": xmin,
            "y2": ymax,
            "x2": xmax
        })

    doc = raw.get("document", {})
    return {
        "image_width": doc.get("image_width"),
        "image_height": doc.get("image_height"),
        "questions": questions
    }
# =======================
# =======================
