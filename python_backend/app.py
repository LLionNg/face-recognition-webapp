import time
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from modules.models_manager import ModelsManager
from modules.image_utils import base64_to_cv2, bytes_to_cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class ImageRequest(BaseModel):
    image: str
    threshold: Optional[float] = 0.5  # Lower default threshold

class FaceResult(BaseModel):
    bbox: List[float]
    confidence: float  # MTCNN detection confidence
    name: str
    recognition_confidence: float  # FaceNet recognition confidence
    class_id: Optional[int] = None

class DetectionResponse(BaseModel):
    faces: List[FaceResult]
    processing_time: float
    threshold_used: float
    pipeline: str

# Initialize models manager
models_manager = ModelsManager(
    facenet_path="models/best_facenet_model.pt",
    confidence_threshold=0.5
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    await models_manager.load_models()
    logger.info("=" * 60)
    logger.info("Face Recognition API Started")
    logger.info("Pipeline: MTCNN (detect & crop) → FaceNet (recognize)")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down Face Recognition API")

app = FastAPI(
    title="Face Recognition API",
    description="MTCNN + FaceNet Pipeline (Training-Matched)",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:6111"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_models_manager():
    return models_manager

@app.post("/detect-and-recognize", response_model=DetectionResponse)
async def detect_and_recognize(
    request: ImageRequest, 
    manager: ModelsManager = Depends(get_models_manager)
):
    """
    Detect faces with MTCNN and recognize with FaceNet.
    Uses EXACT same pipeline as training (prepare_dataset.py + facenet_inference.py)
    """
    start_time = time.time()
    threshold = request.threshold if request.threshold is not None else 0.5
    
    face_detector, face_recognizer, facenet_model, idx_to_class = manager.get_models()
    
    if not face_detector:
        raise HTTPException(status_code=500, detail="Face detector not loaded")
    if not face_recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer not loaded")
    
    try:
        logger.info("Processing new image...")
        image = base64_to_cv2(request.image)
        logger.info(f"Image shape: {image.shape}")
        
        # Detect and crop faces using MTCNN
        logger.info("Step 1: Detecting faces with MTCNN...")
        cropped_faces, bboxes, probs = face_detector.detect_and_crop_faces(image)
        
        logger.info(f"MTCNN found {len(cropped_faces)} face(s)")
        
        faces = []
        
        # Recognize each cropped face with FaceNet
        for i, (face_img, bbox, prob) in enumerate(zip(cropped_faces, bboxes, probs)):
            logger.info(f"\nStep 2: Recognizing face {i+1}/{len(cropped_faces)}...")
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            name, recognition_confidence, class_id = face_recognizer.recognize(face_img)
            
            logger.info(f"Face {i+1} result: {name} (class_id={class_id}, confidence={recognition_confidence:.3f})")
            
            faces.append(FaceResult(
                bbox=[float(x1), float(y1), float(width), float(height)],
                confidence=prob,  # MTCNN detection confidence
                name=name,
                recognition_confidence=recognition_confidence,  # FaceNet confidence
                class_id=class_id
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"\nProcessed in {processing_time:.3f}s")
        logger.info(f"Results: {len(faces)} face(s) detected")
        for i, face in enumerate(faces):
            logger.info(f"  Face {i+1}: {face.name} ({face.recognition_confidence*100:.1f}%)")
        
        return DetectionResponse(
            faces=faces, 
            processing_time=processing_time,
            threshold_used=threshold,
            pipeline="MTCNN → FaceNet (exact training match)"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    manager: ModelsManager = Depends(get_models_manager)
):
    """Test endpoint with file upload using exact training pipeline"""
    start_time = time.time()
    
    face_detector, face_recognizer, facenet_model, idx_to_class = manager.get_models()
    
    if not face_detector or not face_recognizer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        cv2_image = bytes_to_cv2(image_data)
        
        # Detect and crop faces using MTCNN
        cropped_faces, bboxes, probs = face_detector.detect_and_crop_faces(cv2_image)
        
        faces = []
        
        for face_img, bbox, prob in zip(cropped_faces, bboxes, probs):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            name, recognition_confidence, class_id = face_recognizer.recognize(face_img)
            
            faces.append({
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "confidence": prob,
                "name": name,
                "recognition_confidence": recognition_confidence,
                "class_id": class_id
            })
        
        processing_time = time.time() - start_time
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "faces": faces,
            "processing_time": processing_time,
            "threshold_used": threshold,
            "pipeline": "MTCNN (detect & crop) → FaceNet (recognize)",
            "message": f"Found {len(faces)} face(s)"
        }
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/test")
async def test_endpoint(manager: ModelsManager = Depends(get_models_manager)):
    """Test endpoint to verify pipeline"""
    face_detector, face_recognizer, facenet_model, idx_to_class = manager.get_models()
    return {
        "status": "API is running",
        "pipeline": "MTCNN (exact training match) → FaceNet",
        "detector": "MTCNN (image_size=160, post_process=True)",
        "mtcnn_loaded": face_detector is not None,
        "facenet_loaded": facenet_model is not None,
        "recognizer_loaded": face_recognizer is not None,
        "num_classes": len(idx_to_class),
        "class_names": list(idx_to_class.values()) if idx_to_class else [],
        "confidence_threshold": manager.confidence_threshold,
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check(manager: ModelsManager = Depends(get_models_manager)):
    """Health check endpoint"""
    _, _, _, idx_to_class = manager.get_models()
    return {
        "status": "healthy",
        "pipeline": "prepare_dataset.py → facenet_inference.py (exact match)",
        "mtcnn_loaded": manager.face_detector is not None,
        "facenet_loaded": manager.facenet_model is not None,
        "recognizer_loaded": manager.face_recognizer is not None,
        "num_classes": len(idx_to_class),
        "classes": list(idx_to_class.values()) if idx_to_class else [],
        "recognition_mode": "classification",
        "confidence_threshold": manager.confidence_threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)