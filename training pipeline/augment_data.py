import os
import cv2
import numpy as np
import logging
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDataAugmenter:
    
    def __init__(self, augment_factor=3, output_suffix='_aug'):
        self.augment_factor = augment_factor
        self.output_suffix = output_suffix
        
        # Define realistic augmentation pipeline using Albumentations
        self.transform = A.Compose([
            # Geometric transformations (subtle)
            A.OneOf([
                A.Rotate(limit=120, p=0.7),  # Small rotation ±15 degrees
                A.Affine(
                    rotate=(-10, 10),
                    scale=(0.95, 1.05),  # Slight scaling
                    translate_percent=(-0.05, 0.05),  # Small translation
                    p=0.3
                ),
            ], p=0.8),
            
            # Lighting and color adjustments (realistic)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,  # ±15% brightness
                    contrast_limit=0.1,     # ±10% contrast
                    p=0.6
                ),
                # A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),  # Adaptive histogram equalization
                # A.RandomGamma(gamma_limit=(1.0, 1.1), p=0.3),  # Corrected subtle gamma correction
            ], p=0.7),
            
            # Color adjustments
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=5,      # Small hue shift
                    sat_shift_limit=15,     # ±15% saturation
                    val_shift_limit=10,     # ±10% value
                    p=0.4
                ),
                A.RGBShift(
                    r_shift_limit=10,       # Small RGB channel shifts
                    g_shift_limit=10,
                    b_shift_limit=10,
                    p=0.3
                ),
            ], p=0.5),
            
            # Noise and blur (very subtle)
            A.OneOf([
                A.GaussNoise(var_limit=(5, 15), p=0.3),  # Light gaussian noise
                A.Blur(blur_limit=2, p=0.2),             # Very light blur
                A.MotionBlur(blur_limit=3, p=0.1),       # Subtle motion blur
            ], p=0.4),
            
        ], p=1.0, save_applied_params=True)
    
    def augment_image_albumentations(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented_images = []
            
            for i in range(self.augment_factor):
                augmented = self.transform(image=image)['image']
                augmented_images.append(augmented)
            
            return augmented_images
            
        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")
            return []
    
    def save_augmented_images(self, augmented_images, original_path, train_dir):
        """Save augmented images to the same directory"""
        base_name = Path(original_path).stem
        extension = Path(original_path).suffix
        
        saved_count = 0
        
        for i, aug_image in enumerate(augmented_images):
            try:
                new_filename = f"{base_name}{self.output_suffix}_{i+1}{extension}"
                new_path = os.path.join(train_dir, new_filename)
                
                if isinstance(aug_image, np.ndarray):
                    # Convert RGB back to BGR for OpenCV saving
                    if len(aug_image.shape) == 3:
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(new_path, aug_image_bgr)
                    else:
                        cv2.imwrite(new_path, aug_image)
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving augmented image {new_filename}: {e}")
        
        return saved_count
    
    def augment_dataset(self, data_dir, preview_mode=False):
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return
        
        logger.info(f"Starting data augmentation in: {data_dir}")
        logger.info(f"Augmentation factor: {self.augment_factor}")
        logger.info(f"Using Albumentations for augmentation")
        
        total_original = 0
        total_augmented = 0
        processed_classes = 0
        
        class_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        for class_name in tqdm(class_dirs, desc="Processing classes"):
            class_path = os.path.join(data_dir, class_name)
            train_path = os.path.join(class_path, 'Train')
            
            if not os.path.exists(train_path) or not os.path.isdir(train_path):
                logger.warning(f"No Train/ directory found for class: {class_name}")
                continue
            
            processed_classes += 1
            logger.info(f"\nProcessing class: {class_name}")

            for file1 in os.listdir(train_path):
                file_path = os.path.join(train_path, file1)

                if not Path(file1).suffix:
                    try:
                        img = cv2.imread(file_path)
                        if img is not None:
                            ext = '.jpg'  # Default to .jpg
                            new_file_path = file_path + ext
                            os.rename(file_path, new_file_path)
                            logger.info(f"Renamed {file1} to {file1 + ext}")
                    except Exception as e:
                        logger.warning(f"Could not read or rename {file1}: {e}")
            
            image_files = [f for f in os.listdir(train_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                logger.warning(f"No images found in {train_path}")
                continue
            
            original_count = len(image_files)
            total_original += original_count
            augmented_count = 0
            
            for image_file in tqdm(image_files, desc=f"Augmenting {class_name}", leave=False):
                image_path = os.path.join(train_path, image_file)
                
                if self.output_suffix in image_file:
                    continue
                
                if preview_mode:
                    logger.info(f"  Would augment: {image_file}")
                    continue
                
                # Apply augmentation
                augmented_images = self.augment_image_albumentations(image_path)
                
                if augmented_images:
                    # Save augmented images
                    saved = self.save_augmented_images(augmented_images, image_path, train_path)
                    augmented_count += saved
                    total_augmented += saved
            
            logger.info(f"  Class {class_name}: {original_count} original → {augmented_count} augmented")
        
        # logger.info("\n" + "=" * 60)
        # logger.info("AUGMENTATION SUMMARY")
        # logger.info("=" * 60)
        # logger.info(f"Processed classes: {processed_classes}")
        # logger.info(f"Original images: {total_original}")
        # logger.info(f"Augmented images created: {total_augmented}")
        # logger.info(f"Total images after augmentation: {total_original + total_augmented}")
        # logger.info(f"Dataset size multiplier: {(total_original + total_augmented) / max(total_original, 1):.1f}x")

def main():
    parser = argparse.ArgumentParser(description='Augment face recognition training data')
    parser.add_argument('--data_dir', '-d', default='known_people_photos',
                       help='Directory containing face images with Train/ subdirectories')
    parser.add_argument('--augment_factor', '-a', type=int, default=3,
                       help='Number of augmented versions per original image (default: 3)')
    parser.add_argument('--suffix', '-s', default='_aug',
                       help='Suffix for augmented image names (default: _aug)')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview mode - show what would be augmented without actually doing it')
    
    args = parser.parse_args()
    
    try:
        import albumentations
    except ImportError:
        logger.error("Albumentations not installed. Install with: pip install albumentations opencv-python numpy")
        return
    
    # Initialize augmenter
    augmenter = FaceDataAugmenter(
        augment_factor=args.augment_factor,
        output_suffix=args.suffix
    )
    
    # Run augmentation
    augmenter.augment_dataset(
        data_dir=args.data_dir,
        preview_mode=args.preview
    )

if __name__ == "__main__":
    main()