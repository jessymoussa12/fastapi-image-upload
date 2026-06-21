from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI()


class InferenceResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float


@app.get("/")
def root():
    return {"message": "FastAPI vision API is running"}


@app.post("/predict", response_model=InferenceResponse)
async def predict_image(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a JPEG or PNG image."
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded. Please upload a valid image."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be opened as a valid image."
        )

    return InferenceResponse(
        filename=file.filename,
        prediction="mock_prediction",
        confidence=0.95
    )
