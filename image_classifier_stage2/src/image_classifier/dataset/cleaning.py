from pathlib import Path

import cv2


VALID_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}


def remove_invalid_images(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    removed: list[Path] = []

    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            valid_extension = image_path.suffix.lower() in VALID_EXTENSIONS
            image = cv2.imread(str(image_path)) if valid_extension else None

            if not valid_extension or image is None:
                print(f"Removing invalid image: {image_path}")
                image_path.unlink()
                removed.append(image_path)

    return removed
