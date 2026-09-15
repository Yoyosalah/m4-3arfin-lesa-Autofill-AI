import json
import os
import tempfile
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from .graph import app_graph


# =======================
# ADDED FOR FLUTTER LINK
# =======================
async def process_session(
    audio: UploadFile = File(...),
    ocr_json: str = Form(...)
):
    try:
        ocr_data = json.loads(ocr_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="ocr_json is not valid JSON")

    audio_bytes = await audio.read()

    original_name = audio.filename or "audio.wav"
    suffix = os.path.splitext(original_name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio:
        tmp_audio.write(audio_bytes)
        audio_path = tmp_audio.name

    try:
        final_state = app_graph.invoke({
            "audio_path": audio_path,
            "ocr_json": ocr_data
        })
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if final_state.get("error_message"):
        return JSONResponse(
            status_code=500,
            content={"error": final_state["error_message"]}
        )

    return JSONResponse(final_state["extracted_data"])
# =======================
# =======================


def main():
    audio_file_path = "/home/youssef-salah/Cellula/NLP/Autofill/audio_recordings/testing_ASR_6.wav"

    with open("/home/youssef-salah/Cellula/NLP/Autofill/input_json/ocr_result_simple.json", "r", encoding="utf-8") as f:
        ocr_mock_data = json.load(f)

    print(f"Starting pipeline...\n")

    final_state = app_graph.invoke({
        "audio_path": audio_file_path,
        "ocr_json": ocr_mock_data
    })

    if final_state.get("error_message"):
        print(f"\n[!] PIPELINE FAILED: {final_state['error_message']}")
    else:
        print("\n================ FINAL JSON OUTPUT ================")
        print(json.dumps(final_state["extracted_data"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
