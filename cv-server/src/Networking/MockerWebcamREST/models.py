from pydantic import BaseModel


class Frame(BaseModel):
    image_base64: str
