#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import logging
from facenet_pytorch import InceptionResnetV1
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FaceRecognitionInference:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.idx_to_class = self.load_model()

        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Classes: {list(self.idx_to_class.values())}")

    def load_model(self):
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        num_classes = checkpoint["num_classes"]
        facenet = InceptionResnetV1(pretrained="vggface2")
        model = nn.Sequential(
            facenet,
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Load with strict=False to ignore pretrained logits
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(self.device)
        model.eval()

        return model, checkpoint.get("idx_to_class", {})

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                confidence = confidence.item()
                predicted_class = predicted_class.item()

                if confidence >= self.confidence_threshold:
                    person_name = self.idx_to_class.get(str(predicted_class), "Unknown")
                else:
                    person_name = "Unknown"

                return {
                    "person": person_name,
                    "confidence": confidence,
                    "class_id": predicted_class,
                }

        except Exception as e:
            logger.error(f"Error: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Face recognition inference")
    parser.add_argument("--model", "-m", required=True, help="Path to model (.pt file)")
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.7, help="Confidence threshold"
    )

    args = parser.parse_args()

    inference = FaceRecognitionInference(args.model, args.confidence)
    result = inference.predict_image(args.image)

    if result:
        print(f"\nPrediction: {result['person']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Class ID: {result['class_id']}")


if __name__ == "__main__":
    main()
