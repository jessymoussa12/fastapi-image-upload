from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.schemas import InferenceResponse
from app.services.vision_service import predict_image_class

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
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

    result = predict_image_class(contents)

    return InferenceResponse(
        filename=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )
