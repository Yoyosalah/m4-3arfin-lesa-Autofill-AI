from fastapi import FastAPI

from OCR.ocr import create_template
from Agent.agent import process_session


app = FastAPI()


# =======================
# OCR ENDPOINT
# =======================
app.post("/create-template")(create_template)


# =======================
# AGENT ENDPOINT
# =======================
app.post("/process-session")(process_session)