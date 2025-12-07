# This code should run only the first time

import os # for managing files and paths
import shutil # for copying/moving files 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # the path of the current Python file
# ex: if the file is in C:/Projects/ArSL_HandGestureNet/src/prepare_data_structure.py, then BASE_DIR = "C:/Projects/ArSL_HandGestureNet/src"

def organize_images_by_class(images_dir, labels_dir, output_dir):
    # ex:
    # image_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/images/"
    # labels_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/labels/"
    # output_dir = "/ArSL_HandGestureNet/src/../data/structured/train_structured"

    # Checks if the images folder and labels folder exist
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    
    # Creates a new output folder
    os.makedirs(output_dir, exist_ok=True) # exist_ok=True --> if the folder already exists, don’t raise error
    
    for img_name in os.listdir(images_dir):
        """
        `os.listdir`: lists all files in the images folder

        the structure of data 
        data/raw/data/unaugmented/416/
            |
            ├──train/
                    ├──images/
                             ├── IMG_20210609_184853_jpg.rf.4d28bfe14f7d136ec0a22d1353bc5577.jpg
                             .
                             .
                             ├── .......jpg

                    ├──labels/
                             ├── IMG_20210609_184853_jpg.rf.4d28bfe14f7d136ec0a22d1353bc5577.txt
                             .
                             .
                             ├── .......txt
            |
            ├──valid/
                    ├── images/
                    ├── labels/
            |
            ├──test/
                    ├── images/
                    ├── labels/

            so `os.listdir(images_dir)` = os.listdir(image_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/images/")
            = ["IMG_20210609_184853_jpg.rf.4d28bfe14f7d136ec0a22d1353bc5577.jpg", ...., "...jpg"]

        """
        # This condition skips any non-image files (filter)
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # image_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/images/ + img_name = "IMG_20210609_184853_jpg.rf.4d28bfe14f7d136ec0a22d1353bc5577.jpg"
        # image_path = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/images/IMG_20210609_184853_jpg.rf.4d28bfe14f7d136ec0a22d1353bc5577.jpg"
        image_path = os.path.join(images_dir, img_name) 
        # os.path.splitext(img_name)[0] --> gets the image name without extension, Then adds .txt
        label_file = os.path.splitext(img_name)[0] + ".txt"
        # label_path = full path of the label file
        label_path = os.path.join(labels_dir, label_file)
        # If the label file is missing, the code warns and continues with the next image
        if not os.path.exists(label_path):
            print(f"Warning: No label found for image {img_name}")
            continue
        # Opens the label file and reads the first line 
        with open(label_path, "r") as f:
            line = f.readline().strip()

        if line == "":
            print(f"Warning: Empty label file for {img_name}")
            continue
        # line content --> "0 0.5444711538461539 0.5961538461538461 0.4182692307692308 0.4266826923076923" 
        # and we need the label only (first number)
        try:
            class_index = int(line.split()[0])
            # after spliting class_index = "0"
        except Exception:
            print(f"Warning: Could not parse label line for {img_name}: {line}")
            continue
        # Creates a folder for each class inside output
        # output_dir = "/ArSL_HandGestureNet/src/../data/structured/train_structured"
        # class_dir =  "/ArSL_HandGestureNet/src/../data/structured/train_structured/ALIF"
        classes = ['ALIF', 'BAA', 'TA', 'THA', 'JEEM', 'HAA', 'KHAA', 'DELL', 'DHELL',
                 'RAA', 'ZAY', 'SEEN', 'SHEEN', 'SAD', 'DAD', 'TAA', 'DHAA', 'AYN',
                 'GHAYN', 'FAA', 'QAAF', 'KAAF', 'LAAM', 'MEEM', 'NOON', 'HA', 'WAW', 'YA']

        class_dir = os.path.join(output_dir, f"{classes[class_index]}")
        os.makedirs(class_dir, exist_ok=True)
        # Copies the original image to the new folder
        shutil.copy(image_path, os.path.join(class_dir, img_name))

    print("✔ Done organizing images.")


# Train data
# BASE_DIR = "/ArSL_HandGestureNet/src"
# image_dir = os.path.join("/ArSL_HandGestureNet/src", "../data/raw/data/unaugmented/416/train/images/")

# image_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/images/"
# labels_dir = "/ArSL_HandGestureNet/src/../data/raw/data/unaugmented/416/train/labels/"
# output_dir = "/ArSL_HandGestureNet/src/../data/structured/train_structured"

# Train data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/train/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/train/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/train_structured")
)

# Valid data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/valid/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/valid/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/valid_structured")
)

# Test data
organize_images_by_class(
    images_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/test/images/"),
    labels_dir = os.path.join(BASE_DIR, "../data/raw/unaugmented/416/test/labels/"),
    output_dir = os.path.join(BASE_DIR, "../data/structured/test_structured")
)
