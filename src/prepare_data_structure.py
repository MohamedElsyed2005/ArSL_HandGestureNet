# This code should run only the first time

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def organize_images_by_class(images_dir, labels_dir, output_dir):

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(images_dir, img_name)
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: No label found for image {img_name}")
            continue

        with open(label_path, "r") as f:
            line = f.readline().strip()

        if line == "":
            print(f"Warning: Empty label file for {img_name}")
            continue

        try:
            class_index = line.split()[0]
        except Exception:
            print(f"Warning: Could not parse label line for {img_name}: {line}")
            continue

        class_dir = os.path.join(output_dir, f"class_{class_index}")
        os.makedirs(class_dir, exist_ok=True)

        shutil.copy(image_path, os.path.join(class_dir, img_name))

    print("✔ Done organizing images.")


# Train data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/train/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/train/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/train_structured")
)

# Valid data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/valid/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/valid/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/valid_structured")
)

# Test Data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/test/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/data/unaugmented/416/test/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/test_structured")
)
