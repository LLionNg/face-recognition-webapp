import os
import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_dataset(source_dir, dest_dir):
    """
    Detects and crops faces from images in source_dir and saves them to dest_dir.
    Maintains the same directory structure, but only processes 'Train/' directories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for face detection: {device}")

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
    )

    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    logger.info(f"Starting to pre-process images from: {source_dir}")

    processed_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(source_dir):
        if Path(root).name.lower() != "train":
            continue

        class_name = Path(root).parent.name

        dest_class_path = os.path.join(dest_dir, class_name)
        dest_train_path = os.path.join(dest_class_path, "Train")
        os.makedirs(dest_train_path, exist_ok=True)

        logger.info(f"Processing class: {class_name}...")

        for filename in tqdm(
            files, desc=f"Cropping faces for {class_name}", leave=False
        ):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                original_path = os.path.join(root, filename)

                try:
                    image = cv2.imread(original_path)
                    if image is None:
                        logger.warning(f"Could not load image: {original_path}")
                        skipped_count += 1
                        continue

                    # Convert BGR to RGB (for MTCNN)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect and crop face
                    face_tensor = mtcnn(image_rgb, save_path=None)

                    if face_tensor is not None:
                        face_np = (
                            face_tensor.permute(1, 2, 0).detach().cpu().numpy() * 127.5
                            + 127.5
                        ).astype(np.uint8)
                        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                        new_filename = Path(filename).stem + "_cropped" + ".jpg"
                        new_path = os.path.join(dest_train_path, new_filename)
                        cv2.imwrite(new_path, face_bgr)
                        processed_count += 1
                    else:
                        skipped_count += 1

                except Exception as e:
                    logger.error(f"Error processing {original_path}: {e}")
                    skipped_count += 1

    # logger.info("=" * 50)
    # logger.info("Dataset Pre-processing Summary")
    # logger.info("=" * 50)
    # logger.info(f"Processed images: {processed_count}")
    # logger.info(f"Skipped images (no face detected): {skipped_count}")
    # logger.info(f"Final dataset saved to: {dest_dir}")
    # logger.info("Note: Only 'Train' directories were processed.")


if __name__ == "__main__":
    source_data_dir = "known_people_photos"
    dest_data_dir = "cropped_photos"
    prepare_dataset(source_data_dir, dest_data_dir)
