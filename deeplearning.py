# ============================================================
# DEEP LEARNING - CNN
# CAT vs DOG IMAGE CLASSIFICATION
# ============================================================
#
# CNN = Convolutional Neural Network
#
# PROJECT WORKFLOW
#   1. Load Image Dataset
#   2. Preprocess Images
#   3. Extract Features Using CNN
#   4. Flatten Feature Maps
#   5. Classify Images
#   6. Train the Model
#   7. Save the Trained Model
#
# MODEL OUTPUT
#   Probability near 0 -> Cat
#   Probability near 1 -> Dog
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "dataset"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 10

MODEL_PATH = "cat_dog_model.keras"


# ============================================================
# 3. LOAD IMAGE DATASET
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    # Resize all images to 128 × 128
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    # Process 32 images per batch
    batch_size=BATCH_SIZE
)


# ------------------------------------------------------------
# 4. DISPLAY CLASS NAMES
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("             DATASET INFORMATION")
print("=" * 50)

print("Class Names:", train_data.class_names)


# ============================================================
# 5. CREATE CNN MODEL
# ============================================================

model = models.Sequential([

    # --------------------------------------------------------
    # INPUT LAYER
    # Image Shape = 128 × 128 × 3
    #
    # Height  = 128 pixels
    # Width   = 128 pixels
    # Channels = 3 (RGB)
    # --------------------------------------------------------

    layers.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    ),


    # --------------------------------------------------------
    # IMAGE NORMALIZATION
    #
    # Original Pixel Values : 0 - 255
    # Normalized Values     : 0 - 1
    # --------------------------------------------------------

    layers.Rescaling(1.0 / 255),


    # --------------------------------------------------------
    # CNN LAYER 1 - FEATURE EXTRACTION
    #
    # Filters      = 32
    # Kernel Size  = 3 × 3
    # Activation   = ReLU
    # --------------------------------------------------------

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),


    # --------------------------------------------------------
    # MAX POOLING LAYER 1
    # Reduce the spatial dimensions of feature maps
    # --------------------------------------------------------

    layers.MaxPooling2D(),


    # --------------------------------------------------------
    # CNN LAYER 2 - DEEPER FEATURE EXTRACTION
    #
    # Filters      = 64
    # Kernel Size  = 3 × 3
    # Activation   = ReLU
    # --------------------------------------------------------

    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),


    # --------------------------------------------------------
    # MAX POOLING LAYER 2
    # Reduce feature-map dimensions
    # --------------------------------------------------------

    layers.MaxPooling2D(),


    # --------------------------------------------------------
    # FLATTEN LAYER
    #
    # Convert 2D feature maps into a 1D vector
    # --------------------------------------------------------

    layers.Flatten(),


    # --------------------------------------------------------
    # DENSE LAYER - CLASSIFICATION FEATURES
    #
    # Neurons    = 128
    # Activation = ReLU
    # --------------------------------------------------------

    layers.Dense(
        units=128,
        activation="relu"
    ),


    # --------------------------------------------------------
    # OUTPUT LAYER - BINARY CLASSIFICATION
    #
    # Output Neurons = 1
    # Activation     = Sigmoid
    #
    # Probability near 0 -> Cat
    # Probability near 1 -> Dog
    # --------------------------------------------------------

    layers.Dense(
        units=1,
        activation="sigmoid"
    )

])


# ============================================================
# 6. COMPILE CNN MODEL
# ============================================================

model.compile(

    # Optimizer updates model weights during training
    optimizer="adam",

    # Loss function for binary classification
    loss="binary_crossentropy",

    # Track training accuracy
    metrics=["accuracy"]

)


# ============================================================
# 7. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\n" + "=" * 50)
print("             CNN MODEL SUMMARY")
print("=" * 50 + "\n")

model.summary()


# ============================================================
# 8. TRAIN CNN MODEL
# ============================================================

print("\n" + "=" * 50)
print("              TRAINING STARTED")
print("=" * 50 + "\n")

history = model.fit(

    train_data,

    # Number of complete passes through the dataset
    epochs=EPOCHS

)


# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

model.save(MODEL_PATH)


# ============================================================
# 10. TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 50)
print("             TRAINING COMPLETED")
print("=" * 50)

print("Model saved successfully!")
print("Saved Model:", MODEL_PATH)

print("\nCNN Cat vs Dog Classification Finished.")
print("=" * 50)