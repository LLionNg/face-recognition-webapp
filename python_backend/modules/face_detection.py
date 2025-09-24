import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from facenet_pytorch import MTCNN

logger = logging.getLogger(__name__)


class FaceDetector:
    """MTCNN-based face detector with exact training settings"""

    def __init__(self, device="cpu"):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,  # Critical
            device=device,
            keep_all=True,  # Return all detected faces
        )
        logger.info(f"MTCNN initialized on {device} with training-matched parameters")

    def detect_and_crop_faces(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[float]]:
        """
        Detect faces and return cropped face images EXACTLY as done during training
        Returns: (cropped_faces, bboxes, probabilities)
        """
        try:
            # Convert BGR to RGB (MTCNN uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # get bounding boxes
            boxes, probs = self.mtcnn.detect(rgb_image)

            if boxes is None:
                logger.info("No faces detected by MTCNN")
                return [], [], []

            logger.info(f"MTCNN detected {len(boxes)} face(s)")

            cropped_faces = []
            bboxes = []
            probabilities = []

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box]

                    # Extract face using MTCNN's internal method (same as training)
                    # This applies the same preprocessing as during dataset preparation
                    face_tensor = self.mtcnn.extract(rgb_image, [box], None)

                    if face_tensor is not None and len(face_tensor) > 0:
                        # Convert tensor from [-1, 1] to [0, 255] (same as prepare_dataset.py)
                        face_np = (
                            face_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
                            * 127.5
                            + 127.5
                        ).astype(np.uint8)

                        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                        cropped_faces.append(face_bgr)
                        bboxes.append((x1, y1, x2, y2))
                        probabilities.append(float(prob))

                        logger.info(
                            f"Face {i+1} extracted: bbox=({x1},{y1},{x2},{y2}), confidence={prob:.3f}"
                        )
                    else:
                        logger.warning(f"Failed to extract face {i+1}")

                except Exception as e:
                    logger.warning(f"Error extracting face {i}: {e}")
                    continue

            return cropped_faces, bboxes, probabilities

        except Exception as e:
            logger.error(f"Error detecting faces: {e}", exc_info=True)
            return [], [], []


class FaceRecognizer:
    """FaceNet-based face recognizer with exact inference settings"""

    def __init__(self, model, idx_to_class: dict, confidence_threshold=0.5):
        self.model = model
        self.idx_to_class = idx_to_class
        self.confidence_threshold = confidence_threshold
        self.device = next(model.parameters()).device

        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logger.info(f"FaceRecognizer initialized with {len(idx_to_class)} classes")
        logger.info(f"Classes: {list(idx_to_class.values())}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def recognize(self, face_image: np.ndarray) -> Tuple[str, float, int]:
        """
        Recognize face using EXACT same inference pipeline as facenet_inference.py
        face_image: BGR image from MTCNN (already preprocessed to 160x160)
        Returns: (name, confidence, class_id)
        """
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                confidence_val = confidence.item()
                predicted_class_id = predicted_class.item()

                # Map class ID to student ID using idx_to_class
                student_id = self.idx_to_class.get(str(predicted_class_id), "Unknown")

                logger.info(
                    f"Prediction - Class ID: {predicted_class_id}, Confidence: {confidence_val:.3f}"
                )
                logger.info(f"Mapped to: {student_id}")

                # Apply threshold
                if confidence_val >= self.confidence_threshold:
                    person_name = student_id
                    logger.info(
                        f"Recognized: {person_name} (confidence {confidence_val:.3f} >= threshold {self.confidence_threshold})"
                    )
                else:
                    person_name = "Unknown"
                    logger.info(
                        f"Low confidence {confidence_val:.3f} < threshold {self.confidence_threshold}, marking as Unknown"
                    )

                return person_name, confidence_val, predicted_class_id

        except Exception as e:
            logger.error(f"Error recognizing face: {e}", exc_info=True)
            return "Unknown", 0.0, -1
