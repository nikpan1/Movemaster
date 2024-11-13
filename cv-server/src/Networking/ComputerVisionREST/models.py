from pydantic import BaseModel


class Frame(BaseModel):
    base64_image: str


class Settings(BaseModel):
    min_detection_confidence: float
    min_tracking_confidence: float
