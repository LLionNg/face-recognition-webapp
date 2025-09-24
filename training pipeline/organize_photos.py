import os
import shutil
import logging
from pathlib import Path
from PIL import Image
import imghdr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_image_file(file_path):
    """Check if a file is actually an image, even without proper extension"""
    try:
        image_type = imghdr.what(file_path)
        if image_type:
            return True
            
        # Fallback to PIL for more comprehensive check
        with Image.open(file_path) as img:
            img.verify()
            return True
    except:
        return False

def fix_missing_extensions(base_directory="known_people_photos"):
    """Fix files that are images but missing .jpg extension"""
    logger.info("Checking for image files with missing extensions...")
    
    if not os.path.exists(base_directory):
        logger.error(f"Base directory not found: {base_directory}")
        return
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                continue
            
            if is_image_file(file_path):
                # Add .jpg extension
                new_file_path = file_path + '.jpg'
                
                # Make sure the new filename doesn't already exist
                counter = 1
                while os.path.exists(new_file_path):
                    base_name = file_path
                    new_file_path = f"{base_name}_{counter}.jpg"
                    counter += 1
                
                try:
                    os.rename(file_path, new_file_path)
                    logger.info(f"Fixed extension: {file} -> {os.path.basename(new_file_path)}")
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Error fixing extension for {file}: {e}")
    
    logger.info(f"Fixed {fixed_count} files with missing extensions")
    return fixed_count

def organize_student_photos(base_directory="known_people_photos"):
    """
    Move all image files from subdirectories to their parent student ID directories
    
    Structure transformation:
    known_people_photos/
    ├── 65020733/
    │   ├── Test/
    │   │   ├── photo1.jpg  -> moves to 65020733/
    │   │   └── photo2.jpg  -> moves to 65020733/
    │   └── Train/
    │       ├── photo3.jpg  -> moves to 65020733/
    │       └── photo4.jpg  -> moves to 65020733/
    """
    
    if not os.path.exists(base_directory):
        logger.error(f"Base directory not found: {base_directory}")
        return
    
    # First, fix any missing extensions
    fix_missing_extensions(base_directory)
    
    total_moved = 0
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    for student_id in os.listdir(base_directory):
        student_path = os.path.join(base_directory, student_id)
        
        if not os.path.isdir(student_path):
            continue
            
        logger.info(f"Processing student ID: {student_id}")
        moved_count = 0
        
        for item in os.listdir(student_path):
            item_path = os.path.join(student_path, item)
            
            if os.path.isdir(item_path):
                logger.info(f"  Found subdirectory: {item}")
                
                for file in os.listdir(item_path):
                    if file.lower().endswith(image_extensions):
                        source_path = os.path.join(item_path, file)
                        
                        destination_file = file
                        destination_path = os.path.join(student_path, destination_file)
                        
                        counter = 1
                        while os.path.exists(destination_path):
                            name, ext = os.path.splitext(file)
                            destination_file = f"{name}_{counter}{ext}"
                            destination_path = os.path.join(student_path, destination_file)
                            counter += 1
                        
                        # Move the file
                        try:
                            shutil.move(source_path, destination_path)
                            logger.info(f"    Moved: {file} -> {destination_file}")
                            moved_count += 1
                            total_moved += 1
                        except Exception as e:
                            logger.error(f"    Error moving {file}: {e}")
                
                # Remove subdirectory if it's empty
                try:
                    remaining_files = os.listdir(item_path)
                    if not remaining_files:
                        os.rmdir(item_path)
                        logger.info(f"    Removed empty directory: {item}")
                    else:
                        logger.warning(f"    Directory {item} not empty, contains: {remaining_files}")
                except Exception as e:
                    logger.error(f"    Error removing directory {item}: {e}")
        
        logger.info(f"  Student {student_id}: {moved_count} files moved")
    
    logger.info(f"Organization complete. Total files moved: {total_moved}")

def verify_organization(base_directory="known_people_photos"):
    """Verify the organization by showing the final structure"""
    logger.info("Final directory structure:")
    
    if not os.path.exists(base_directory):
        logger.error(f"Base directory not found: {base_directory}")
        return
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    for student_id in sorted(os.listdir(base_directory)):
        student_path = os.path.join(base_directory, student_id)
        
        if not os.path.isdir(student_path):
            continue
        
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(image_extensions)]
        subdirs = [f for f in os.listdir(student_path) 
                  if os.path.isdir(os.path.join(student_path, f))]
        other_files = [f for f in os.listdir(student_path) 
                      if not f.lower().endswith(image_extensions) and 
                      not os.path.isdir(os.path.join(student_path, f))]
        
        logger.info(f"{student_id}/")
        logger.info(f"  Image files: {len(image_files)}")
        if image_files:
            for img in sorted(image_files):
                logger.info(f"    - {img}")
        
        if other_files:
            logger.info(f"  Other files: {other_files}")
        
        if subdirs:
            logger.info(f"  Remaining subdirectories: {subdirs}")

def fix_extensions_only(base_directory="known_people_photos"):
    """Only fix missing extensions without moving files"""
    logger.info("Running extension fix only...")
    fix_missing_extensions(base_directory)
    verify_organization(base_directory)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize student photos by moving images to parent directories')
    parser.add_argument('--directory', '-d', default='known_people_photos', 
                       help='Base directory containing student folders (default: known_people_photos)')
    parser.add_argument('--verify-only', '-v', action='store_true',
                       help='Only verify current structure without moving files')
    parser.add_argument('--fix-extensions-only', '-f', action='store_true',
                       help='Only fix missing extensions without organizing')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_organization(args.directory)
    elif args.fix_extensions_only:
        fix_extensions_only(args.directory)
    else:
        organize_student_photos(args.directory)
        verify_organization(args.directory)