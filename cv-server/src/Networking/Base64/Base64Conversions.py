import base64
import cv2
import numpy as np

def base64_to_image(input_str: str) -> cv2.Mat:
    '''
    Function to convert base64 string to image file
    '''
    img_data = base64.b64decode(input_str)
    np_array = np.frombuffer(img_data, np.uint8)
    image_mat = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return image_mat

def image_to_base64(image: cv2.Mat) -> str:
    ''' Function to convert cv2 Mat to base64 string '''
    retval, buffer = cv2.imencode('.jpg', image)
    base64_string = base64.b64encode(buffer)

    return base64_string