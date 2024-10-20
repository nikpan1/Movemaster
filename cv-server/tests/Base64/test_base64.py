import base64
import cv2
from src.Base64.Base64Conversions import *


IMAGE_FILENAME = "image.jpg"

def isBase64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s
    except Exception:
        return False

def load_image() -> cv2.Mat:
    image = cv2.imread(IMAGE_FILENAME)
    return image

def load_image_b64() -> str:
    image = cv2.imread(IMAGE_FILENAME)
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)

    return jpg_as_text


# --------------------------

def test_base64_to_image():
    image = load_image_b64()

    output = base64_to_image(image)

    width, height, _ = output.shape
    assert (width > 0)
    assert (height > 0)


def test_image_to_base64():
    image_b64 = load_image()

    b64_str = image_to_base64(image_b64)

    assert (isBase64(b64_str))
