# This code should run only the first time

import os
import shutil

def organize_images_by_class(images_dir, labels_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        image_path = os.path.join(images_dir, img_name)
        label_file = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            line = f.readline().strip()
            class_index = line.split()[0]

        class_dir = os.path.join(output_dir, f"class_{class_index}")
        os.makedirs(class_dir, exist_ok=True)

        shutil.copy(image_path, os.path.join(class_dir, img_name))

# Train data
organize_images_by_class(
    images_dir="data/unaugmented/416/train/images/",
    labels_dir="data/unaugmented/416/train/labels/",
    output_dir="data/unaugmented/416/train/out_put/"
)

# Valid data
organize_images_by_class(
    images_dir="data/unaugmented/416/valid/images/",
    labels_dir="data/unaugmented/416/valid/labels/",
    output_dir="data/unaugmented/416/valid/out_put/"
)

# Test Data
organize_images_by_class(
    images_dir="data/unaugmented/416/test/images/",
    labels_dir="data/unaugmented/416/test/labels/",
    output_dir="data/unaugmented/416/test/out_put/"
)
