import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rename_files_without_extension(root_dir):
    if not os.path.isdir(root_dir):
        logger.error(f"Directory not found: {root_dir}")
        return

    logger.info(f"Starting to scan for files without extensions in: {root_dir}")

    renamed_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            # Check if the file has no extension
            base, ext = os.path.splitext(filename)
            if not ext:
                original_path = os.path.join(dirpath, filename)
                new_path = original_path + ".jpg"

                try:
                    os.rename(original_path, new_path)
                    logger.info(f"Renamed '{original_path}' to '{new_path}'")
                    renamed_count += 1
                except Exception as e:
                    logger.error(f"Error renaming '{original_path}': {e}")

    logger.info("=" * 50)
    logger.info(f"File renaming complete. Renamed {renamed_count} files.")
    logger.info("=" * 50)


if __name__ == "__main__":
    data_directory = "known_people_photos"
    rename_files_without_extension(data_directory)
