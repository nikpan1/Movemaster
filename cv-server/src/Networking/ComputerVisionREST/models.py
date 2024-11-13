from pydantic import BaseModel


class Frame(BaseModel):
    image_base64: str


class Settings(BaseModel):
    detection_confidence: float
    tracking_confidence: float
