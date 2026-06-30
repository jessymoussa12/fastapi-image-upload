from pydantic import BaseModel


class InferenceResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
