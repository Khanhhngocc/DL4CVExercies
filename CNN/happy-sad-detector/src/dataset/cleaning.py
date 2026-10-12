import os

import cv2
from PIL import Image


IMAGE_EXTS = [
    "jpeg",
    "jpg",
    "bmp",
    "png",
]


def remove_invalid_images(data_dir):
    removed_images = []

    for image_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, image_class)

        if not os.path.isdir(class_path):
            continue

        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)

            try:
                # Check the actual image format
                with Image.open(image_path) as img:

                    img_format = (
                        img.format.lower()
                        if img.format
                        else ""
                    )

                    # Check whether the image file is corrupted
                    img.verify()

                # Check supported formats
                if img_format not in IMAGE_EXTS:
                    print(
                        f"Unsupported image format "
                        f"({img_format}): "
                        f"{image_path}"
                    )

                    os.remove(image_path)
                    removed_images.append(image_path)

                    continue

                # Check whether OpenCV can decode it
                cv_img = cv2.imread(image_path)

                if cv_img is None:

                    print(f"OpenCV cannot read: " f"{image_path}")

                    os.remove(image_path)
                    removed_images.append(image_path)

            except Exception as e:
                print(f"Invalid image " f"{image_path}: {e}")

                if os.path.exists(image_path):
                    os.remove(image_path)
                    removed_images.append(image_path)

    return removed_images