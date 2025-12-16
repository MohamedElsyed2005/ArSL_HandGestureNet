"""
Image Preprocessing Pipeline for Hand Gesture Recognition
========================================================

What this script does:
- Takes raw hand gesture images from the dataset
- Cleans them up (removes noise, fixes lighting problems)
- Crops to focus on the hand region
- Resizes everything to 224x224.

WHY we do each step is explained in the comments below!

Input:  data/structured/
                ├── train_structured/
                ├── valid_structured
                ├── test_structured

Output: data/processed/
                ├── train
                ├── valid 
                ├── test

No explicit hand segmentation is applied in this pipeline
Segmentation was intentionally excluded due to instability and inconsistency across real-world conditions
Instead, a safe enhancement + center crop strategy is used to maintain dataset integrity and training stability

This preprocessing is applied consistently to train, validation, and test 
datasets to avoid data leakage.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # creates a progress bar

# =========================
# CONFIG
# =========================
IMG_SIZE = 224

# =========================
# Utility function to safely create directories
# =========================
def ensure_dir(p):
    """Create folder if it doesn't exist. Won't crash if folder already there."""
    os.makedirs(p, exist_ok=True)

# =========================
# SAFE IMAGE ENHANCER
# =========================
class SafeEnhancer:
    """
    This class makes our images look better for the CNN to learn from
    
    We tried multiple approaches and settled on this 3-step process:
        1. Remove noise (while preserving edges), which is why we chose the Bilateral filter
        2. Fix lighting problems (CLAHE)
        3. Make edges slightly stronger (Laplacian & wieghts)
    
    IMPORTANT: We do NOT do segmentation (cutting out just the hand)
    because we tested it and it failed too often on different backgrounds/lighting
    """
    def enhance(self, img):
        """
        Takes a raw image and returns a cleaned-up version
        
        Input: img (BGR image from cv2.imread)
        Output: enhanced image (same size, same BGR format)
        """
        # =====================================================================
        # STEP 1: Remove noise without blurring edges
        # =====================================================================
        # Problem: Camera sensors add random noise to images
        # Normal blur (like Gaussian) would make the hand edges blurry, or (like Median blur) distorts shapes
        # 
        # Solution: Bilateral filter
        # - It removes noise in flat areas (like background)
        # - But KEEPS edges sharp (like finger outlines)
        # - Think of it as "smart blur"
        #
        # Parameters we chose:
        # - d=5: How big the blur area is (5 pixels)
        # - 40, 40: How much to blur (tried 20, 30, 40, 60 - this worked best)
        img = cv2.bilateralFilter(img, 5, 40, 40)

        # =====================================================================
        # STEP 2: Fix lighting problems
        # =====================================================================
        # Problem: Some images are too dark, some too bright, shadows everywhere
        # This confuses the CNN because the same hand looks different
        #
        # Solution: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # We do this in LAB color space because:
        # - LAB separates brightness (L) from colors (A and B)
        # - We only fix brightness, keep skin color the same
        # - Prevents weird color shifts
        
        # Convert BGR → LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # "Adaptive" = it works on small 8x8 patches, not the whole image
        # "Contrast Limited" = clipLimit=2.0 prevents over-brightening
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l) # Only fix the brightness channel

        # Merge channels back and convert to BGR
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # =====================================================================
        # STEP 3: Make edges slightly stronger
        # =====================================================================
        # Problem: After denoising, some edges are a bit soft
        # CNNs learn better when they can clearly see finger boundaries
        #
        # Solution: Add a tiny bit of edge information back
        
        # First, find all the edges using Laplacian
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=3) # Find edges 
        """
        Why cv2.CV_16S?
        The Laplacian operator computes second-order derivatives, which can produce negative values
        CV_16S (16-bit signed integer) can store both positive and negative values,
        preserving all edge information correctly
        """
        edges = cv2.convertScaleAbs(edges) # Convert to normal image format
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Make it 3-channel

        # Blend edges softly to avoid over-sharpening
        img = cv2.addWeighted(img, 1.0, edges, 0.15, 0)
        """Important: edges weight = 0.15 (very light), avoids harsh sharpening while improving boundary clarity for the CNN"""
        # output = img1 * α + img2 * β + γ
        #==================================================================
        # img1  : The main (original) image
        # img2  : The secondary image (here: edge map)
        #
        # α :
        #   - Weight of the original image
        #   - α = 1.0 means keep the original image fully visible
        #
        # β :
        #   - Weight of the edge image
        #   - Small values (0.1 – 0.2) softly enhance boundaries
        #   - Prevents harsh sharpening or noise
        #
        # γ :
        #   - Constant bias added to all pixels
        #   - Usually set to 0 (no brightness shift)
        #
        # In this case:
        #   img = img * 1.0 + edges * 0.15
        #   → Original image is preserved
        #   → Edges are gently reinforced to make finger boundaries clearer
        #   → Ideal for CNN training (better shape & contour learning)
        # =====================================================================
        return img

# =========================
# CENTER CROP (CRITICAL STEP)
# =========================
def center_crop(img, size=IMG_SIZE):
    """
    Cut out the center part of the image where the hand usually is
    
    Why do this?
    - Our dataset has hands mostly in the center
    - Removes distracting background stuff (walls, furniture, etc.)
    - Makes the CNN focus on the hand, not the room
    
    The 85% number:
    - We tested different percentages:
    - 70%: Cut off fingers sometimes 
    - 80%: Still cut fingers occasionally 
    - 85%: Never cut fingers, removes most background (WE USE THIS)
    - 90%: Safe but keeps too much background 
    
    Input: img (any size)
    Output: cropped and resized to 224x224
    """
    h, w = img.shape[:2] # Get image height and width

    # Calculate how big our crop should be
    # We use 85% of the smaller dimension (height or width)
    crop_size = int(min(h, w) * 0.85)

    # Find the center point
    cx, cy = w // 2, h // 2

    # Calculate top-left corner of our crop box
    x1 = max(cx - crop_size // 2, 0) # max() prevents going outside image
    y1 = max(cy - crop_size // 2, 0)

    # Cut out the center square
    crop = img[y1:y1+crop_size, x1:x1+crop_size]

    # Resize to exactly 224x224
    crop = cv2.resize(crop, (size, size))
    return crop

# =========================
# DATASET PROCESSING FUNCTION
# =========================
def process_dataset(input_dir, output_dir):
    """
    Go through all images in a folder and process them.
    
    Folder train_structured:
            ALIF/
                image1.jpg
                image2.jpg
            BAA/
                image1.jpg
                ...
    
    For each image:
    1. Load it
    2. Enhance it (SafeEnhancer)
    3. Crop it (center_crop)
    4. Save it to output_dir
    """
    ensure_dir(output_dir)
    enh = SafeEnhancer() # Create output folder for this class


    log = [] # Keep track of what we processed (for debugging later)

    # Loop through each class folder (ALIF, BAA, etc.)
    for cls in tqdm(os.listdir(input_dir), desc="Classes"):
        in_cls = os.path.join(input_dir, cls)
        out_cls = os.path.join(output_dir, cls)

        # Skip if it's not a folder
        if not os.path.isdir(in_cls):
            continue

        ensure_dir(out_cls) # Create output folder for this class

        # Process each image in this class folder
        for img_name in os.listdir(in_cls):
            if not img_name.lower().endswith(('.jpg','.png','.jpeg')): # Only process image files
                continue

            img_path = os.path.join(in_cls, img_name)
            img = cv2.imread(img_path)

            # Skip if image is corrupted or can't be read
            if img is None:
                continue

            # STEP 1: Enhance the image
            img_e = enh.enhance(img)

            # STEP 2: Crop to center
            img_final = center_crop(img_e)

            # STEP 3: Save processed image
            cv2.imwrite(os.path.join(out_cls, img_name), img_final)

            # Keep a log entry
            log.append({
                "class": cls,
                "image": img_name,
                "action": "enhanced_only"
            })

    # Save the log as a CSV file
    pd.DataFrame(log).to_csv(os.path.join(output_dir,"log.csv"), index=False)

    print(f"Processed images saved to: {output_dir}")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    """This preprocessing is applied consistently to train, validation, and test datasets to avoid data leakage"""
    SPLITS = {
        "train": "train_structured",
        "valid": "valid_structured",
        "test":  "test_structured"
    }

    # Get the project root directory
    BASE = os.path.dirname(os.path.dirname(__file__))

    # Process each split one by one
    for split, folder in SPLITS.items():
        INPUT = os.path.join(BASE, "data", "structured", folder)
        OUTPUT = os.path.join(BASE, "data", "processed", split)

        print(f"\n--- Processing {split.upper()} ---")
        process_dataset(INPUT, OUTPUT)

    print("\n ALL SPLITS PROCESSED (ENHANCE ONLY)")


"""
===========================================
NOTE ON THE PROBLEMS WE FACED
===========================================

WHY NO SEGMENTATION?
--------------------

We tried to segment the hand (cut out just the hand pixels) using:
- Skin detection using (HSV/RGB/YCbCr)
- Morphological filtering 
- Largest contour detection 
- then crop the hand region (usually largest contour)

Results: ALL FAILED too often! 

Problems:
- Failed when hand was close to same-color object
- Failed on dark backgrounds
- Failed with shadows
- Failed on different skin tones

So we gave up on segmentation and just do enhancement + center crop
This approach is significantly more reliable

===========================================
FINAL NOTE
===========================================
This preprocessing pipeline is designed to be stable and consistent,
instead of using aggressive image processing

Each step was selected through testing to help the CNN learn better
and work well on different real-world images,
while avoiding techniques like hand segmentation that were not reliable
"""