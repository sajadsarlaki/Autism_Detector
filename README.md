# ASD Image Classification using VGG16 and Keras Tuner

This project applies deep learning techniques to classify facial images as either "Autistic" or "Non-Autistic" using transfer learning with VGG16. The model is trained on a labeled dataset and evaluated through k-fold cross-validation and final testing.

## Project Motivation

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that impacts communication and behavior. Early and accessible detection is vital for timely support and intervention. This project explores the use of computer vision and transfer learning to assist in ASD screening based on facial image features.

## Model Overview

- **Base Model:** VGG16 (pretrained on ImageNet)
- **Added Layers:** Flatten → Dense → Dropout → Sigmoid Output
- **Hyperparameter Tuning:** Conducted using Keras Tuner (Random Search)
- **Training Technique:** K-Fold Cross-Validation with EarlyStopping and ModelCheckpoint
- **Final Evaluation:** Performance metrics computed on a held-out test set

## Dataset Structure

The dataset must be structured as follows:

dataset/ ├── train/ │ ├── Autistic/ │ └── Non_Autistic/ ├── valid/ │ ├── Autistic/ │ └── Non_Autistic/ ├── test/ │ ├── Autistic/ │ └── Non_Autistic/


Images should be in standard formats such as `.jpg` or `.png`, and placed in the appropriate folder.

## Key Features

- **Image Preprocessing:** Includes resizing, normalization, and augmentation (flip, rotation, brightness/contrast, CLAHE, etc.) using Albumentations.  
- **Model Tuning:** Optimizes learning rate, dense layer size, and dropout rate using Keras Tuner.  
- **Cross-Validation:** 3-fold cross-validation to ensure model generalization.  
- **Metrics Used:** Accuracy, Precision, Recall, F1 Score, and AUC-ROC.  

## How to Run

1. Install required packages:
    ```bash
    pip install tensorflow keras keras-tuner albumentations opencv-python scikit-learn
    ```

2. Prepare your dataset according to the structure shown above.

3. Set the correct path to your dataset:
    ```python
    dataset_dir = '/path/to/your/AutismDataset'
    ```

4. Run `project.py`. This will:
    - Load and preprocess the data  
    - Perform hyperparameter tuning  
    - Conduct k-fold training  
    - Evaluate on the test set  
    - Save the final trained model as `autism_final_model.h5`  

## Results

Average performance from k-fold cross-validation and final test evaluation:

- **Accuracy:** _[Insert your result]_  
- **Precision:** _[Insert your result]_  
- **Recall:** _[Insert your result]_  
- **F1 Score:** _[Insert your result]_  
- **AUC-ROC:** _[Insert your result]_  

(Replace these with your actual values printed at the end of `project.py`)

## Authors

- [Your Name]  
- [Teammate's Name]  

## Disclaimer

This project is intended for academic and research purposes only. It is not intended for clinical diagnosis or real-world medical use.

