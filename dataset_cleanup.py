# ============================================================
# CNN DATASET CLEANUP
# ============================================================
# CNN = Convolutional Neural Network
#
# A Deep Learning architecture commonly used for:
#   - Image Classification
#   - Object Detection
#   - Face Recognition
#
# DATASET CLEANUP TASKS
#   1. Validate image files
#   2. Delete empty or corrupted images
#   3. Convert valid images to RGB
#   4. Save cleaned images for CNN training
#
# Pillow Installation:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
# ============================================================

import os
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\rahul\Desktop\Codes\Cat vs Dog Classification\dataset"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
)


# ============================================================
# 2. INITIALIZE COUNTERS
# ============================================================

total = 0
cleaned = 0
deleted = 0


# ============================================================
# 3. START DATASET CLEANUP
# ============================================================

print("\n" + "=" * 50)
print("          CNN DATASET CLEANUP STARTED")
print("=" * 50)


# ============================================================
# 4. SCAN AND VALIDATE IMAGES
# ============================================================

for root, folders, files in os.walk(DATASET_PATH):

    for filename in files:

        # Skip files that are not supported image formats
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        file_path = os.path.join(root, filename)
        total += 1

        try:

            # ------------------------------------------------
            # STEP 1: CHECK FOR EMPTY FILES
            # ------------------------------------------------

            if os.path.getsize(file_path) == 0:

                print("\n[DELETE] Empty image:")
                print(file_path)

                os.remove(file_path)
                deleted += 1

                continue

            # ------------------------------------------------
            # STEP 2: OPEN AND VALIDATE IMAGE
            # ------------------------------------------------

            with Image.open(file_path) as image:

                # Fully load the image to detect corrupted files
                image.load()

                original_mode = image.mode

                # ------------------------------------------------
                # STEP 3: CONVERT IMAGE TO RGB
                # ------------------------------------------------

                if original_mode != "RGB":

                    print(f"\n[CONVERT] {original_mode} -> RGB")
                    print(file_path)

                    image = image.convert("RGB")

                # ------------------------------------------------
                # STEP 4: SAVE CLEANED IMAGE
                # ------------------------------------------------

                if filename.lower().endswith((".jpg", ".jpeg")):

                    image.save(
                        file_path,
                        format="JPEG",
                        quality=95
                    )

                else:

                    image.save(file_path)

                cleaned += 1

        # ----------------------------------------------------
        # STEP 5: HANDLE CORRUPTED OR UNREADABLE IMAGES
        # ----------------------------------------------------

        except Exception as error:

            print("\n[ERROR] Invalid image detected:")
            print(file_path)
            print("Reason:", error)

            try:

                os.remove(file_path)

                print("[DELETE] Corrupted image removed.")
                deleted += 1

            except OSError as delete_error:

                print("[FAILED] Could not delete image:")
                print(delete_error)


# ============================================================
# 5. DISPLAY CLEANUP SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("             CLEANUP COMPLETED")
print("=" * 50)

print(f"Total images checked : {total}")
print(f"Images cleaned       : {cleaned}")
print(f"Images deleted       : {deleted}")

print("\nDataset cleanup process finished.")
print("=" * 50)