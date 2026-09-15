from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class ClinicalField(BaseModel):
    input_id: str = Field(description="The exact fillable_id provided in the prompt (e.g., 'input_001').")
    field_name: str = Field(description="e.g., 'Chief Complaint', 'Diagnosis', 'Medications'")
    answer: str = Field(description="The extracted information.")
    confidence: str = Field(description="Must be: 'High', 'Medium', 'Low', or 'Missing'")
    references: List[str] = Field(description="Exact, word-for-word quotes from the transcription.")

class ClinicalDocument(BaseModel):
    fields: List[ClinicalField]

class AppState(TypedDict):
    audio_path: str           
    ocr_json: dict             
    parsed_fields: List[dict]  
    transcription: str        
    extracted_data: dict      
    errors: List[str]         
    revision_count: int
    error_message: Optional[str]