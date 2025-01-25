# Table of Contents

1. [Features](#features)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Folder Structure](#folder-structure)
3. [Pipeline Overview](#pipeline-overview)
    - [Step 1: Feature Extraction](#step-1-feature-extraction)
    - [Step 2: Save Dataset in Multiple Formats](#step-2-save-dataset-in-multiple-formats)
    - [Step 3: Train the Neural Network](#step-3-train-the-neural-network)
    - [Step 4: Save the Model](#step-4-save-the-model)
    - [Step 5: Predict Classes for New Images](#step-5-predict-classes-for-new-images)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Example Predictions](#example-predictions)
6. [Notes and Recommendations](#notes-and-recommendations)
7. [Acknowledgments](#acknowledgments)

---

# PersianFace Project

This project aims to build a face recognition pipeline using the **DeepFace** library and the **ArcFace** model. It processes a dataset of facial images, extracts features, trains a Multi-Layer Perceptron (MLP) neural network, and allows predictions for new images. The pipeline is designed for efficient feature extraction and classification of facial images.

---

## Features

- **Face Embedding Extraction**: Utilizes the **ArcFace** model in DeepFace to extract 512-dimensional feature vectors from facial images.
- **Dataset Support**: Converts image datasets into feature vectors and saves them in multiple formats (CSV, JSON, NPY).
- **Custom Neural Network**: Implements and trains an MLP neural network for face recognition.
- **Model Persistence**: Saves trained models for reuse and real-time predictions.
- **Batch Predictions**: Supports bulk processing and predictions for new images.
- **Error Handling**: Manages missing faces or unreadable images gracefully.

---

## Getting Started

### Prerequisites

Install the required libraries:
```bash
pip install deepface tensorflow pandas numpy tqdm matplotlib
```

### Folder Structure
Make sure your project follows this structure:
```
.
|-- PersianFace/
    |-- Image_Folder/
        |-- Class_1/
            |-- image1.jpg
            |-- image2.jpg
        |-- Class_2/
            |-- image1.jpg
            |-- image2.jpg
    |-- output/
        |-- PersianFace_features.csv
        |-- mlp_model.h5
```

---

## Pipeline Overview

### Step 1: Feature Extraction
Extract features using the ArcFace model and save them into a CSV file.
```python
generate_dataset_with_arcface(image_folder="Image_Folder", output_file_path="output/PersianFace_features.csv")
```

### Step 2: Save Dataset in Multiple Formats
Save the extracted dataset in `.csv`, `.json`, and `.npy` formats.
```python
save_dataset(df, csv_path="dataset.csv", json_path="dataset.json", npy_path="dataset.npy")
```

### Step 3: Train the Neural Network
Train an MLP model on the extracted dataset.
```python
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
```

### Step 4: Save the Model
Save the trained model for reuse.
```python
model.save("mlp_model.h5")
```

### Step 5: Predict Classes for New Images
Load the trained model and predict the class for new images.
```python
predicted_label = model.predict(np.array([embedding]))
predicted_class = le.inverse_transform([np.argmax(predicted_label)])
```

---

## Evaluation Metrics

| Metric          | Value  |
|------------------|--------|
| Train Loss       | 0.279  |
| Train Accuracy   | 92.4%  |
| Test Loss        | 0.464  |
| Test Accuracy    | 89%  |

---

## Example Predictions

For images in the folder `new_images_folder`, the predictions may look like this:
```
File: Elnaz.jpg, Predicted Class: Elnaz_Shakerdoost
File: Mohammad-Reza_Shajarian.jpg, Predicted Class: Mohammad_Reza_Shajarian
File: Sahar.jpg, Predicted Class: Sahar_Dolatshahi
File: Tarane.jpg, Predicted Class: Tarane_Alidoosti
File: Tarane2.jpg, Predicted Class: Tarane_Alidoosti
File: Sahab.jpg, Predicted Class: Shahab_Hoseini
File: Shahab2.jpg, Predicted Class: Shahab_Hoseini
File: Parinaz.jpg, Predicted Class: Parinaz_Izadyar
```

---

## Notes and Recommendations

1. **Face Detection Errors**:
   - If some faces are not detected, set `enforce_detection=False` in `DeepFace.represent`.

2. **Data Quality**:
   - Ensure images are high-quality and contain only a single face.

3. **Model Tuning**:
   - Hyperparameters such as learning rate, number of neurons, and epochs can be fine-tuned for better results.

4. **Batch Predictions**:
   - Use the `predict_images_with_labels` function to process and classify multiple images in one go.

---

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for facial recognition and feature extraction.
- TensorFlow/Keras for building and training the neural network.
- Pandas and Numpy for data manipulation and storage.

---

This version includes a table of contents for better navigation. You can directly copy this into your `README.md`. Let me know if you need further assistance!