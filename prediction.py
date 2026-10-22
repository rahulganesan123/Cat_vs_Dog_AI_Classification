# ============================================================
# DEEP LEARNING - CNN
# CAT vs DOG IMAGE PREDICTION
# ============================================================
#
# WORKFLOW
#   1. Load Trained Model
#   2. Read Test Images
#   3. Load and Resize Images
#   4. Convert Images to NumPy Arrays
#   5. Add Batch Dimension
#   6. Predict Using CNN
#   7. Classify as Cat or Dog
#   8. Display Prediction Results
#
# OUTPUT
#   Probability < 0.5  -> Cat
#   Probability >= 0.5 -> Dog
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import tensorflow as tf
import numpy as np


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_PATH = "cat_dog_model.keras"

TEST_FOLDER = "test_images"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 50)
print("          CAT vs DOG PREDICTION")
print("=" * 50)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nTrained model loaded successfully!")


# ============================================================
# 4. VALIDATE TEST IMAGE FOLDER
# ============================================================

if not os.path.isdir(TEST_FOLDER):

    raise FileNotFoundError(
        f"Test folder not found: {TEST_FOLDER}"
    )


# ============================================================
# 5. READ AND PROCESS TEST IMAGES
# ============================================================

total_images = 0

for filename in sorted(os.listdir(TEST_FOLDER)):

    # --------------------------------------------------------
    # PROCESS ONLY SUPPORTED IMAGE FILES
    # --------------------------------------------------------

    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue

    image_path = os.path.join(
        TEST_FOLDER,
        filename
    )

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD AND RESIZE IMAGE
        # ----------------------------------------------------

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            color_mode="rgb"
        )


        # ----------------------------------------------------
        # STEP 2: CONVERT IMAGE TO NUMPY ARRAY
        # ----------------------------------------------------

        image = tf.keras.utils.img_to_array(image)


        # ----------------------------------------------------
        # STEP 3: ADD BATCH DIMENSION
        #
        # Shape before: (128, 128, 3)
        # Shape after : (1, 128, 128, 3)
        # ----------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        )


        # ----------------------------------------------------
        # STEP 4: PREDICT IMAGE CLASS
        # ----------------------------------------------------

        prediction = model.predict(
            image,
            verbose=0
        )[0][0]


        # ----------------------------------------------------
        # STEP 5: CLASSIFY AS CAT OR DOG
        # ----------------------------------------------------

        if prediction < 0.5:

            result = "Cat"
            confidence = (1 - prediction) * 100

        else:

            result = "Dog"
            confidence = prediction * 100


        # ----------------------------------------------------
        # STEP 6: DISPLAY PREDICTION RESULT
        # ----------------------------------------------------

        print("\nImage     :", filename)
        print("Prediction:", result)
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Output: {prediction:.4f}")

        total_images += 1

    except Exception as error:

        print("\n[ERROR] Could not process image:")
        print("Image:", filename)
        print("Reason:", error)


# ============================================================
# 6. DISPLAY PREDICTION SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("             PREDICTION COMPLETED")
print("=" * 50)

print("Total images predicted:", total_images)

print("\nCat vs Dog prediction process finished.")
print("=" * 50)