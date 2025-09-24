import torch
import torch.nn as nn
import os
import json
import logging
from facenet_pytorch import InceptionResnetV1
from modules.face_detection import FaceDetector, FaceRecognizer

logger = logging.getLogger(__name__)

class ModelsManager:
    """Manages FaceNet model and MTCNN detector with exact training pipeline"""
    
    def __init__(self, facenet_path: str, confidence_threshold: float = 0.5):
        self.facenet_model = None
        self.face_detector = None
        self.face_recognizer = None
        self.idx_to_class = {}
        self.facenet_path = facenet_path
        self.confidence_threshold = confidence_threshold
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    async def load_models(self):
        """Load FaceNet model and initialize MTCNN detector"""
        try:
            logger.info(f"Loading FaceNet model from {self.facenet_path}")
            self.facenet_model, self.idx_to_class = self.load_facenet_from_checkpoint(self.facenet_path)
            logger.info("FaceNet model loaded successfully")
            
            logger.info("Initializing MTCNN face detector...")
            self.face_detector = FaceDetector(device=self.device)
            logger.info("MTCNN face detector initialized successfully")
            
            if self.idx_to_class:
                self.face_recognizer = FaceRecognizer(
                    self.facenet_model, 
                    self.idx_to_class, 
                    self.confidence_threshold
                )
                logger.info(f"Face recognizer initialized with {len(self.idx_to_class)} classes")
                logger.info(f"Classes available: {list(self.idx_to_class.values())}")
            else:
                logger.error("No class mappings found in model checkpoint!")
                raise RuntimeError("Model checkpoint missing idx_to_class mapping")

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise RuntimeError(f"Application startup failed: {e}")

    def load_facenet_from_checkpoint(self, model_path):
        """
        Load FaceNet model EXACTLY as in facenet_inference.py
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        num_classes = checkpoint.get('num_classes')
        if num_classes is None:
            raise ValueError("Checkpoint missing 'num_classes' field")
        
        logger.info(f"Model trained with {num_classes} classes")
        
        facenet = InceptionResnetV1(pretrained='vggface2')
        model = nn.Sequential(
            facenet,
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Load weights with strict=False (same as inference)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        idx_to_class = checkpoint.get('idx_to_class', {})
        idx_to_class = {str(k): str(v) for k, v in idx_to_class.items()}

        if not idx_to_class:
            json_path = model_path.replace('.pt', '_mappings.json')
            logger.info(f"No idx_to_class in checkpoint, trying {json_path}")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    idx_to_class = json.load(f).get('idx_to_class', {})
                logger.info(f"Loaded from JSON: {len(idx_to_class)} classes")
            else:
                logger.error(f"JSON file not found: {json_path}")

        logger.info(f"Final idx_to_class: {idx_to_class}")
        return model, idx_to_class

    def get_models(self):
        """Returns loaded models and recognizer"""
        return self.face_detector, self.face_recognizer, self.facenet_model, self.idx_to_class
    
    def get_device(self):
        """Returns the device being used"""
        return self.device