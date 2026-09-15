import os
from google import genai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .schemas import AppState, ClinicalDocument
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

llm = ChatOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-3.5-flash-lite",
)
structured_llm = llm.with_structured_output(ClinicalDocument)

# Prompt
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert clinical document filler. Extract information for the following fields:\n"
               "{fields_to_extract}\n\n"
               "CRITICAL RULES:\n"
               "1. You MUST include the exact 'input_id' for each field as provided above.\n"
               "2. Your 'references' MUST be a list of exact, word-for-word substrings from the transcription.\n"
               "3. If the information is scattered, add each separate quote as a new string in the list. NEVER use ellipses (...) to combine quotes.\n"
               "4. If a requested field is not mentioned, mark confidence as 'Missing' and leave references empty."),
    ("user", "Transcription:\n{transcription}\n\n{error_injection}")
])

# Nodes
def ocr_parsing_node(state: AppState):
    """Node that parses the OCR JSON to map text labels to their fillable input IDs"""

    print("-> Running OCR Parsing Node...")
    try:
        ocr_data = state.get("ocr_json", {})
        
        text_regions = {item["id"]: item["text"].replace("\n", " ") for item in ocr_data.get("text_regions", [])}
        
        parsed_fields = []
        # Match labels to input boxes based on the field_mappings array
        for mapping in ocr_data.get("field_mappings", []):
            if mapping.get("fillable_id") is not None:
                text_id = mapping.get("text_id")
                label = text_regions.get(text_id, "Unknown Field")
                parsed_fields.append({
                    "input_id": mapping.get("fillable_id"),
                    "label": label
                })
        
        if not parsed_fields:
            return {"error_message": "No fillable fields found in the OCR JSON."}
            
        return {"parsed_fields": parsed_fields}
    except Exception as e:
        return {"error_message": f"OCR Parsing Failed: {str(e)}"}

def stt_node(state: AppState):
    """node that converts uploaded audio into text for later field extraction"""

    print("-> Running STT Node...")
    
    if not os.path.exists(state["audio_path"]):
        return {"error_message": "Local audio file not found."}

    audio_file = None
    try:
        audio_file = gemini_client.files.upload(file=state["audio_path"])
        stt_prompt = (
            "Transcribe this Egyptian Arabic audio verbatim, but write all medical "
            "terminology, acronyms, and medications in English (Latin script)."
        )
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[stt_prompt, audio_file]
        )
        return {"transcription": response.text.strip(), "revision_count": 0, "errors": [], "error_message": None}
    except Exception as e:
        return {"error_message": f"STT Generation Failed: {str(e)}"}
    finally:
        # guaranteed execution to maintain 0GB storage constraint
        if audio_file:
            try:
                gemini_client.files.delete(name=audio_file.name)
            except Exception as e:
                print(f"Warning: Failed to delete remote file {audio_file.name}: {e}")

def extraction_node(state: AppState):
    """node that fetches required clinical information from the transcription and structures them"""

    print(f"-> Running Extraction Node (Revision {state.get('revision_count', 0)})...")
    
    # Format the parsed OCR fields into a clear string for the LLM instructions
    fields_list = [f"- ID: {f['input_id']} | Label: '{f['label']}'" for f in state.get("parsed_fields", [])]
    fields_to_extract = "\n".join(fields_list)
    
    error_injection = ""
    if state.get("errors"):
        error_injection = "CRITICAL ERRORS TO FIX:\n" + "\n".join([f"- {err}" for err in state["errors"]])
        error_injection += "\nRewrite the references to EXACTLY match the transcription."

    formatted_prompt = extraction_prompt.format_messages(
        fields_to_extract=fields_to_extract,
        transcription=state["transcription"],
        error_injection=error_injection
    )
    
    try:
        result = structured_llm.invoke(formatted_prompt)
        return {"extracted_data": result.model_dump(), "revision_count": state.get("revision_count", 0) + 1}
    except Exception as e:
        return {"error_message": f"Structured Parsing Failed: {str(e)}"}

def validation_node(state: AppState):
    """node performing validation on input_id and references to check whether extracted quotes match the transcript and flags inconsistencies"""
    print("-> Running Validation Node...")
    
    extracted_fields = state.get("extracted_data", {}).get("fields", [])
    transcription = state.get("transcription", "").lower()
    
    # Grab the valid IDs we generated from the OCR parsing step
    parsed_fields = state.get("parsed_fields", [])
    valid_input_ids = {f["input_id"] for f in parsed_fields}
    
    errors = []

    for field in extracted_fields:
        # Validate the input_id mapping
        current_id = field.get("input_id")
        if current_id not in valid_input_ids:
            errors.append(
                f"Mapping Error in '{field.get('field_name')}': ID '{current_id}' is invalid. "
                f"You must strictly use one of the provided 'input_id' values."
            )
            
        if field.get("confidence") == "Missing":
            continue

        # Validate the transcription references for hallucination prevention
        references = field.get("references", [])
        for ref in references:
            clean_ref = ref.lower().strip()
            if clean_ref in ["n/a", ""]:
                continue

            if clean_ref not in transcription:
                errors.append(f"Hallucination in '{field.get('field_name')}': Quote '{clean_ref}' does not exist.")

    return {"errors": errors}

# Conditions
def should_loop(state: AppState):
    """determine  whether the workflow should retry extraction or stop according to validation results"""
    if len(state.get("errors", [])) > 0 and state.get("revision_count", 0) < 3:
        return "rewrite"
    return "finish"

def check_fatal_error(state: AppState):
    """short-circuit router that stops the graph if an API/System error occurs."""
    if state.get("error_message"):
        return "finish"
    return "continue"