import base64
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import HTTPException

def base64_to_cv2(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv2_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to OpenCV image"""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv2_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")