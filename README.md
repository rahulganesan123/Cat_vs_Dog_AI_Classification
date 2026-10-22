# Cat vs Dog Image Classification Using CNN — GitHub Documentation

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into two categories: **Cat** and **Dog**.

The project contains three Python programs:

1. **Dataset Cleanup** — validates images, converts them to RGB, and removes files that cannot be processed.
2. **CNN Model Training** — loads the dataset, builds the CNN architecture, trains the model, and saves it.
3. **Image Prediction** — loads the trained model and predicts whether new images contain a cat or a dog.

This documentation explains the purpose, workflow, and important code sections of all three programs.

---

## 1. Project Overview

### What is a CNN?

A Convolutional Neural Network (CNN) is a Deep Learning architecture commonly used for image-related tasks, such as image classification, object detection, and face recognition.

Unlike traditional machine learning approaches that often require manually designed image features, a CNN learns useful visual patterns directly from training images.

For this project, the model learns patterns that help distinguish cats from dogs, such as shapes, textures, ears, eyes, and other visual characteristics.

### Project Workflow

1. **Dataset Cleanup**
   - Validate image files.
   - Convert images to RGB.
   - Remove empty or corrupted images.
2. **Load Dataset**
   - Read images from the `cats/` and `dogs/` folders.
3. **Preprocess Images**
   - Resize images to `128 × 128`.
   - Normalize pixel values from `0–255` to `0–1`.
4. **Extract Features**
   - Apply convolutional layers and ReLU activation.
   - Reduce feature-map dimensions using max pooling.
5. **Classify Images**
   - Flatten feature maps.
   - Pass features through dense layers.
   - Use a sigmoid output for binary classification.
6. **Train and Save**
   - Train for 10 epochs.
   - Save the model as `cat_dog_model.keras`.
7. **Predict New Images**
   - Load the saved model.
   - Classify images from the `test_images/` folder.

---

## 2. Project Folder Structure

Organize the project approximately as follows:

```text
Cat vs Dog Classification/
│
├── dataset/
│   ├── cats/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   │
│   └── dogs/
│       ├── dog1.jpg
│       ├── dog2.jpg
│       └── ...
│
├── test_images/
│   ├── test_cat.jpg
│   ├── test_dog.jpg
│   └── ...
│
├── dataset_cleanup.py
├── cnn_training.py
├── prediction.py
├── cat_dog_model.keras
└── README.md
```

The Python filenames above are suggested names; use the actual filenames you choose for your scripts.

- `dataset/` contains labeled training images organized by class.
- `test_images/` contains images you want to classify after training.
- `dataset_cleanup.py` cleans the dataset.
- `cnn_training.py` trains the CNN model.
- `prediction.py` predicts the classes of new images.
- `cat_dog_model.keras` stores the trained model.
- `README.md` documents the project.

**Important:** Run the scripts from the project directory, or configure the paths accordingly. The training and prediction scripts must be able to find the same saved model file.

---

## 3. Program 1 — Dataset Cleanup

### Purpose

Before training a CNN, the dataset should contain images that can be opened and processed correctly. This program scans the dataset folders and performs basic cleanup.

### Main Tasks

- Identify supported image files.
- Delete empty files.
- Open and fully load each image to detect many corrupted files.
- Convert images to RGB format.
- Save the cleaned images.
- Display a summary of the cleanup process.

### 3.1 Import Libraries

```python
import os
from PIL import Image
```

- `os` provides functions to navigate folders, list files, check file sizes, construct file paths, and delete files.
- `PIL.Image` comes from the Pillow library and provides functions to open, process, convert, and save images.

### 3.2 Configure the Dataset Path

```python
DATASET_PATH = r"C:\Users\rahul\Desktop\Codes\Cat vs Dog Classification\dataset"
```

This variable specifies the main dataset directory.

The `r` before the string creates a raw string, allowing Windows backslashes to be used without treating them as escape sequences.

The script uses `os.walk()` to scan the directory and its subdirectories, including `cats/` and `dogs/`.

### 3.3 Define Supported Image Formats

```python
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
)
```

This tuple lists the image extensions that the script attempts to process.

```python
if not filename.lower().endswith(IMAGE_EXTENSIONS):
    continue
```

- `filename.lower()` converts the filename to lowercase.
- `endswith()` checks whether the filename ends with a supported extension.
- `continue` skips files that do not match.

This prevents unrelated files, such as `.txt` or `.csv`, from being processed as images.

### 3.4 Initialize Counters

```python
total = 0
cleaned = 0
deleted = 0
```

These variables track the cleanup results.

| Variable | Purpose |
|---|---|
| `total` | Number of supported image files checked |
| `cleaned` | Number of images successfully processed and saved |
| `deleted` | Number of images successfully deleted |

### 3.5 Scan the Dataset

```python
for root, folders, files in os.walk(DATASET_PATH):
    for filename in files:
        ...
```

`os.walk()` traverses the dataset directory recursively.

- `root` is the current directory path.
- `folders` contains the subdirectory names.
- `files` contains the filenames in the current directory.

The inner `for` loop processes each file individually.

```python
file_path = os.path.join(root, filename)
total += 1
```

`os.path.join()` constructs the complete file path. The counter increases for each supported image encountered.

### 3.6 Detect Empty Files

```python
if os.path.getsize(file_path) == 0:
    os.remove(file_path)
    deleted += 1
    continue
```

`os.path.getsize()` returns the file size in bytes.

If the size is zero, the file contains no data and cannot be a valid image. The program deletes it and skips to the next file.

### 3.7 Open and Validate Images

```python
with Image.open(file_path) as image:
    image.load()
```

- `Image.open()` opens the image file.
- `image.load()` forces Pillow to load the image data, helping detect corrupted or unreadable images.
- `with` ensures the image file is closed when the block finishes.

An image that opens successfully is not necessarily guaranteed to be suitable for every future operation, but this is a useful initial validation step.

### 3.8 Convert Images to RGB

```python
original_mode = image.mode

if original_mode != "RGB":
    image = image.convert("RGB")
```

Images can have different color modes, including:

| Mode | Meaning |
|---|---|
| `RGB` | Red, Green, Blue |
| `RGBA` | Red, Green, Blue, Alpha transparency |
| `L` | Grayscale |
| `P` | Indexed palette color |

The model expects three color channels because its input shape is `(128, 128, 3)`.

`image.convert("RGB")` converts the image into a three-channel RGB representation.

This provides consistent color-channel formatting for the training pipeline. Transparency is not preserved when converting to RGB.

### 3.9 Save the Cleaned Image

```python
if filename.lower().endswith((".jpg", ".jpeg")):
    image.save(
        file_path,
        format="JPEG",
        quality=95
    )
else:
    image.save(file_path)

cleaned += 1
```

For JPEG images, the program explicitly saves the file as JPEG with quality set to 95.

For other supported formats, Pillow infers the output format from the filename extension.

The `cleaned` counter increases after a successful save.

**Important:** This program overwrites the original images. Back up the dataset before running it. Also, some formats such as animated GIFs may lose animation or other format-specific properties during conversion.

### 3.10 Handle Corrupted Images

```python
except Exception as error:
    print("\n[ERROR] Invalid image detected:")
    print(file_path)
    print("Reason:", error)

    try:
        os.remove(file_path)
        deleted += 1
    except OSError as delete_error:
        print("[FAILED] Could not delete image:")
        print(delete_error)
```

If processing an image raises an exception, the program attempts to delete that file.

If deletion also fails, the error is displayed.

This is useful for identifying problematic files, but it is destructive: a valid image that fails during saving or conversion could also be deleted. Review errors and retain a backup.

### 3.11 Display the Cleanup Summary

```python
print(f"Total images checked : {total}")
print(f"Images cleaned       : {cleaned}")
print(f"Images deleted       : {deleted}")
```

The summary provides a quick overview of the cleanup results.

The intended outcome is a dataset containing readable images in a consistent RGB format, ready for the CNN training script.

---

## 4. Program 2 — CNN Model Training

### Purpose

This program creates and trains a CNN using TensorFlow and Keras.

The model receives cat and dog images, extracts useful visual features, and learns to predict which class an image belongs to.

### 4.1 Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import layers, models
```

- `tensorflow` provides the Deep Learning framework.
- `layers` contains neural network components such as convolutional, pooling, and dense layers.
- `models` provides tools for creating and managing models.

### 4.2 Configure the Model

```python
DATASET_PATH = "dataset"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 10

MODEL_PATH = "cat_dog_model.keras"
```

These variables control the training process.

| Parameter | Value | Purpose |
|---|---:|---|
| `IMAGE_HEIGHT` | 128 | Image height in pixels |
| `IMAGE_WIDTH` | 128 | Image width in pixels |
| `BATCH_SIZE` | 32 | Images processed per batch |
| `EPOCHS` | 10 | Complete passes through the training dataset |
| `MODEL_PATH` | `cat_dog_model.keras` | Output model filename |

### 4.3 Load the Image Dataset

```python
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)
```

`image_dataset_from_directory()` loads images from subdirectories and automatically assigns labels based on their directory names.

Expected structure:

```text
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

The function:

1. Reads images from the class folders.
2. Resizes them to 128 × 128 pixels.
3. Assigns labels to the classes.
4. Groups the images into batches of 32.

Each image has the shape `(128, 128, 3)` after loading.

A batch of 32 images has the shape `(32, 128, 128, 3)`.

### 4.4 Display Class Names

```python
print("Class Names:", train_data.class_names)
```

This displays the class names detected from the directory structure.

By default, the class names are ordered alphabetically. For the folders `cats` and `dogs`, the expected order is:

```python
['cats', 'dogs']
```

The associated labels are therefore typically:

- `cats` → `0`
- `dogs` → `1`

This matters because the output layer produces a probability that is interpreted using these labels.

### 4.5 Create the CNN Model

```python
model = models.Sequential([
    ...
])
```

`Sequential` builds a model by stacking layers in order. Data passes through each layer before reaching the output.

#### Layer 1 — Input

```python
layers.Input(shape=(128, 128, 3))
```

This specifies the shape of one image.

- Height: 128 pixels
- Width: 128 pixels
- Channels: 3 for RGB

The batch dimension is added automatically when the model processes a batch of images.

#### Layer 2 — Image Normalization

```python
layers.Rescaling(1.0 / 255)
```

An image's pixel values typically range from 0 to 255.

The rescaling layer divides every pixel value by 255:

`Normalized Pixel = Original Pixel / 255`

Examples:

| Original pixel | Normalized pixel |
|---:|---:|
| 0 | 0.000 |
| 127.5 | 0.500 |
| 255 | 1.000 |

This places the pixel values in the range 0 to 1.

Normalization is performed inside the model, so the prediction program should pass image arrays in the original 0–255 range rather than divide them by 255 again.

#### Layer 3 — First Convolutional Layer

```python
layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu"
)
```

`Conv2D` applies learnable filters to the image to extract visual features.

- `filters=32`: the layer learns 32 filters.
- `kernel_size=(3, 3)`: each filter examines a 3 × 3 region at a time.
- `activation="relu"`: applies the Rectified Linear Unit activation function.

ReLU is defined as `ReLU(x) = max(0, x)`.

It converts negative activation values to zero while retaining positive values.

Early convolutional layers may learn simple patterns such as edges, corners, and textures.

#### Layer 4 — First Max-Pooling Layer

```python
layers.MaxPooling2D()
```

Max pooling reduces the spatial dimensions of the feature maps by selecting the maximum value within local regions.

With the default 2 × 2 pool and stride of 2, it approximately halves the height and width.

For this architecture, the spatial dimensions progress as follows:

| Stage | Output shape per image |
|---|---|
| Input | `128 × 128 × 3` |
| First Conv2D | `126 × 126 × 32` |
| First MaxPooling2D | `63 × 63 × 32` |

The default convolution padding is `valid`, which means the convolution does not add padding around the image borders.

#### Layer 5 — Second Convolutional Layer

```python
layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
)
```

This layer learns 64 filters and processes the feature maps generated by the first convolutional layer.

As the network becomes deeper, it can learn more complex combinations of features rather than only simple visual patterns.

For example, it may learn combinations of edges, textures, and shapes that help distinguish cats from dogs.

#### Layer 6 — Second Max-Pooling Layer

```python
layers.MaxPooling2D()
```

This reduces the spatial dimensions again.

| Stage | Output shape per image |
|---|---|
| Second Conv2D | `61 × 61 × 64` |
| Second MaxPooling2D | `30 × 30 × 64` |

The resulting feature maps are smaller than the original image while retaining learned information.

#### Layer 7 — Flatten

```python
layers.Flatten()
```

The convolutional layers produce three-dimensional feature maps for each image: height, width, and channels.

`Flatten` converts these feature maps into a one-dimensional vector.

For this model:

`30 × 30 × 64 = 57,600`

Therefore, the flattened representation contains 57,600 values per image.

The vector is passed to the dense layer for classification.

#### Layer 8 — Dense Layer

```python
layers.Dense(
    units=128,
    activation="relu"
)
```

A dense layer connects each input value to the layer's neurons.

- `units=128`: the layer contains 128 neurons.
- `activation="relu"`: introduces non-linearity.

This layer learns combinations of the features extracted by the convolutional layers and uses them to help distinguish the two classes.

#### Layer 9 — Output Layer

```python
layers.Dense(
    units=1,
    activation="sigmoid"
)
```

This layer produces one value between 0 and 1 using the sigmoid activation function.

`Sigmoid(x) = 1 / (1 + exp(-x))`

The model interprets the output as the probability estimate for class `1`, which corresponds to dogs under the expected alphabetical class ordering.

| Output | Predicted class |
|---:|---|
| 0.12 | Cat |
| 0.35 | Cat |
| 0.50 | Dog under the script's threshold rule |
| 0.78 | Dog |
| 0.96 | Dog |

These values are model probability estimates, not guarantees of correctness.

### 4.6 Compile the Model

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Compiling configures how the model learns and how its performance is measured.

**Optimizer — Adam**

Adam adjusts the model's weights during training to reduce the loss function.

**Loss — Binary cross-entropy**

Binary cross-entropy measures the difference between the predicted probability and the actual binary label.

For a binary classification problem, it is suitable for a single sigmoid output.

**Metric — Accuracy**

Accuracy measures the proportion of predictions that match the actual labels.

For example, if 80 out of 100 training images are classified correctly, the accuracy is 80%.

### 4.7 Display Model Architecture

```python
model.summary()
```

This displays:

- Layer names and types
- Output shapes
- Number of trainable and non-trainable parameters
- Total parameter count

It is useful for checking that the model architecture was constructed as expected.

### 4.8 Train the Model

```python
history = model.fit(
    train_data,
    epochs=EPOCHS
)
```

`model.fit()` begins the training process.

During each epoch, the model:

1. Receives a batch of images and labels.
2. Performs a forward pass to calculate predictions.
3. Calculates the loss.
4. Computes gradients through backpropagation.
5. Updates the weights using the optimizer.
6. Repeats the process for the remaining batches.

After completing all batches, one epoch is finished. The process repeats for 10 epochs.

The returned `history` object stores training metrics, such as loss and accuracy, for each epoch.

**Training limitation:** The current code trains on the entire dataset supplied to `image_dataset_from_directory()` and does not create a separate validation or test set. Training accuracy alone does not tell you how well the model generalizes to unseen images.

### 4.9 Save the Trained Model

```python
model.save(MODEL_PATH)
```

This saves the trained model as:

```text
cat_dog_model.keras
```

The saved file contains the model architecture, learned weights, and training configuration needed for standard Keras model loading.

The prediction program loads this file instead of training the CNN again.

---

## 5. Program 3 — Cat vs Dog Image Prediction

### Purpose

The prediction program uses the previously trained model to classify new images stored in the `test_images/` directory.

It does not train the model. It loads the saved model, processes each supported image, obtains a prediction, and displays the result.

### 5.1 Import Libraries

```python
import os
import tensorflow as tf
import numpy as np
```

- `os` accesses the test folder and builds image paths.
- `tensorflow` loads the model and processes images.
- `numpy` adds the batch dimension to each image array.

### 5.2 Configure File Paths

```python
MODEL_PATH = "cat_dog_model.keras"
TEST_FOLDER = "test_images"
```

The model path identifies the saved CNN model.

The test folder contains the images to classify.

The model file and test folder must be accessible from the directory in which the script runs, unless absolute paths or another path configuration is used.

### 5.3 Load the Trained Model

```python
model = tf.keras.models.load_model(MODEL_PATH)
```

`load_model()` restores the previously saved Keras model.

This allows the program to use the learned weights immediately without repeating the training process.

If the model file cannot be found, loading will fail.

### 5.4 Validate the Test Folder

```python
if not os.path.isdir(TEST_FOLDER):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_FOLDER}"
    )
```

`os.path.isdir()` checks whether the specified path exists and is a directory.

If the folder is missing, `FileNotFoundError` stops execution and displays a clear error message.

### 5.5 Read Test Images

```python
for filename in sorted(os.listdir(TEST_FOLDER)):

    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue
```

`os.listdir()` returns the entries inside the test directory.

`sorted()` processes filenames in alphabetical order, making the output more predictable.

The extension check filters the entries to supported image files: `.jpg`, `.jpeg`, and `.png`.

Unlike the cleanup script, the prediction script does not process every image format listed in the cleanup program.

### 5.6 Build the Image Path

```python
image_path = os.path.join(
    TEST_FOLDER,
    filename
)
```

This creates the full path to the image.

For example:

```text
test_images/test_cat.jpg
```

### 5.7 Load and Resize the Image

```python
image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="rgb"
)
```

The image is loaded and resized to 128 × 128 pixels.

`color_mode="rgb"` ensures that the loaded image has three color channels.

Using the same image dimensions and color format as training helps maintain consistency between the training and prediction pipelines.

### 5.8 Convert the Image to a NumPy Array

```python
image = tf.keras.utils.img_to_array(image)
```

The loaded image is converted into a numerical array.

Its shape is:

```text
(128, 128, 3)
```

The array contains the pixel values that the CNN needs to process.

At this point, pixel values are normally in the 0–255 range. The model's `Rescaling` layer converts them to 0–1 automatically.

### 5.9 Add a Batch Dimension

```python
image = np.expand_dims(
    image,
    axis=0
)
```

The model expects a batch of images, even when predicting just one image.

Before adding the batch dimension:

```text
(128, 128, 3)
```

After adding the batch dimension:

```text
(1, 128, 128, 3)
```

The first dimension represents one image in the batch.

### 5.10 Generate a Prediction

```python
prediction = model.predict(
    image,
    verbose=0
)[0][0]
```

`model.predict()` passes the image through the trained CNN and returns the output probability.

- `[0]` selects the first image's prediction.
- `[0]` selects the single output value for that image.

The final result is a scalar probability between 0 and 1.

`verbose=0` suppresses the prediction progress display.

### 5.11 Classify the Image

```python
if prediction < 0.5:
    result = "Cat"
    confidence = (1 - prediction) * 100
else:
    result = "Dog"
    confidence = prediction * 100
```

The program uses `0.5` as its decision threshold.

- If the output is less than 0.5, it predicts `Cat`.
- If the output is greater than or equal to 0.5, it predicts `Dog`.

For the confidence display, it uses the probability associated with the predicted class.

**Example:** If the model output is `0.82`, the predicted class is Dog and the displayed score is 82%.

If the output is `0.18`, the predicted class is Cat and the displayed score is 82%, calculated as `1 - 0.18`.

This score should not be interpreted as a guarantee that the prediction is correct. The model can be confidently wrong, especially on unfamiliar images or when training data is limited.

### 5.12 Display the Result

```python
print("\nImage     :", filename)
print("Prediction:", result)
print(f"Confidence: {confidence:.2f}%")
print(f"Raw Output: {prediction:.4f}")
```

The program displays four pieces of information:

| Output | Meaning |
|---|---|
| `Image` | Name of the test image |
| `Prediction` | Predicted class: Cat or Dog |
| `Confidence` | Displayed probability for the predicted class |
| `Raw Output` | Original sigmoid output from the model |

The `:.2f` format displays two decimal places, while `:.4f` displays four decimal places.

### 5.13 Handle Image-Processing Errors

```python
except Exception as error:
    print("\n[ERROR] Could not process image:")
    print("Image:", filename)
    print("Reason:", error)
```

If an individual image cannot be loaded or processed, the program prints the filename and error message.

The loop can then continue with the next image rather than stopping at the first problematic file.

The `total_images` counter increases only after a prediction completes successfully.

### 5.14 Display the Prediction Summary

```python
print("Total images predicted:", total_images)
```

At the end, the program displays how many images were successfully classified.

This makes it easy to check whether all intended test images were processed.

---

## 6. Important Concepts Used in This Project

| Concept | Where it is used | Purpose |
|---|---|---|
| File handling | Dataset cleanup and prediction | Read, save, and remove files |
| Exception handling | All three programs | Handle errors during processing |
| RGB conversion | Dataset cleanup | Standardize image color channels |
| Image resizing | Training and prediction | Standardize image dimensions |
| Normalization | CNN training model | Scale pixel values to 0–1 |
| Convolution | CNN model | Learn visual features |
| ReLU | CNN model | Introduce non-linearity |
| Max pooling | CNN model | Reduce feature-map dimensions |
| Flatten | CNN model | Convert feature maps to a vector |
| Dense layers | CNN model | Learn classification patterns |
| Sigmoid | Output layer | Produce a binary probability estimate |
| Binary cross-entropy | Model compilation | Measure prediction error |
| Adam optimizer | Model compilation | Update model weights |
| Epochs and batches | Training | Organize the learning process |
| Model serialization | Model saving | Reuse the trained model |
| NumPy arrays | Prediction | Represent images as numerical data |

---

## 7. Installation and Execution

### Install the required libraries

```bash
python -m pip install tensorflow pillow numpy
```

### Run the programs in order

```bash
python dataset_cleanup.py
```

```bash
python cnn_training.py
```

```bash
python prediction.py
```

Replace these example script names with your actual filenames if they differ.

The cleanup program modifies the dataset, the training program creates `cat_dog_model.keras`, and the prediction program uses that saved model to classify images from `test_images/`.

---

## 8. Limitations and Future Improvements

The current project is a basic CNN implementation intended to demonstrate the complete image-classification workflow. Several improvements could make it more reliable.

- **Validation dataset:** Split the dataset into training and validation sets to monitor performance on images not used for weight updates.
- **Test dataset:** Evaluate the final model on a separate test set to estimate generalization.
- **Data augmentation:** Apply transformations such as flips, small rotations, or zooms to improve robustness.
- **Dropout or regularization:** Consider techniques that reduce overfitting.
- **Performance metrics:** Track precision, recall, F1-score, and a confusion matrix in addition to accuracy.
- **Model architecture:** Experiment with additional convolutional layers or transfer learning using a pretrained image model.
- **Dataset safety:** Back up the original images before cleanup because the cleanup script overwrites and deletes files.

---

## 9. Conclusion

This project demonstrates an end-to-end image-classification pipeline using Python, TensorFlow, Keras, Pillow, and NumPy.

The first program prepares the dataset, the second builds and trains a CNN, and the third loads the saved model to classify new images as cats or dogs.

It provides practical experience with image preprocessing, convolutional neural networks, binary classification, model training, saving and loading models, and inference on unseen images.

The project can serve as a foundation for exploring more advanced computer vision tasks and improving model accuracy through validation, augmentation, and transfer learning.
