
# Image Classification Using ResNet-18 Transfer Learning

## Introduction

This project implements image classification using **ResNet-18 and Transfer Learning** on the Intel Image Classification dataset. The model classifies images into six scene categories: buildings, forest, glacier, mountain, sea, and street.

Two approaches are explored:
- **Feature Extraction:** Freezing pretrained layers and training only the final classification layer.
- **Fine-Tuning:** Training the `layer4` block along with the final classification layer.

## Objectives

- Understand transfer learning using a pretrained CNN.
- Implement ResNet-18 for image classification.
- Apply image preprocessing and data augmentation.
- Compare feature extraction and fine-tuning.
- Evaluate the model using accuracy, loss, precision, recall, and F1-score.
- Visualize training curves, confusion matrix, and incorrectly classified images.

## Dataset

**Dataset:** Intel Image Classification

The dataset contains six classes:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

### Dataset Structure

```text
image-classification/
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
└── seg_pred/
    └── seg_pred/
```

The training dataset is divided into training and validation sets. The test dataset is used for final evaluation.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- Transfer Learning
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebooks

## Methodology

### 1. Data Loading and Preprocessing

Images are loaded using PyTorch's `ImageFolder`. Preprocessing includes resizing, tensor conversion, normalization, and data augmentation for training images.

### 2. Feature Extraction

A pretrained ResNet-18 model with ImageNet weights is used. The pretrained layers are frozen, and the final fully connected layer is replaced with a new layer for six-class classification.

Only the final classification layer is trained for 3 epochs.

**Final Validation Accuracy: 92.55%**

### 3. Fine-Tuning

The `layer4` block and final classification layer are trained for 3 additional epochs, allowing the model to adapt its higher-level features to the scene classification task.

**Final Validation Accuracy: 93.30%**

### 4. Model Evaluation

The final model is evaluated using:

- Test accuracy
- Test loss
- Classification report
- Confusion matrix
- Incorrectly classified image visualization
- Softmax probabilities

## Results

### Model Performance

| Metric | Result |
|---|---:|
| Feature Extraction Validation Accuracy | 92.55% |
| Fine-Tuning Validation Accuracy | 93.30% |
| Test Accuracy (reported) | 93.03% |
| Test Loss (reported) | 0.2016 |

*Note: Confirm the reported test metrics by rerunning evaluation with the correctly configured six-class test dataset.*

### Performance Comparison

Fine-tuning increased validation accuracy from **92.55% to 93.30%**, an improvement of approximately **0.75 percentage points**.

This suggests that adapting the deeper ResNet-18 layer helped the model learn features more suitable for scene classification.

## Key Insights

- **Transfer learning is effective:** Pretrained ResNet-18 achieved strong validation accuracy without training the entire network from scratch.
- **Fine-tuning improved performance:** Validation accuracy increased by approximately 0.75 percentage points after fine-tuning.
- **Pretrained features are useful:** ImageNet-trained features provide a strong starting point for classifying natural scenes.
- **Evaluation matters:** Accuracy alone does not explain every model error, so the confusion matrix and classification report provide additional insights.
- **Incorrect predictions are informative:** Reviewing misclassified images can help identify visually similar classes and areas for improvement.

## Visualizations

The notebook includes:

- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Incorrectly classified images
- Predicted class labels and softmax probabilities

## How to Run the Project

1. Open the notebook in Kaggle.
2. Add the Intel Image Classification dataset.
3. Verify the training and test directory paths.
4. Run the data preprocessing and data loader cells.
5. Train the feature extraction model for 3 epochs.
6. Fine-tune `layer4` for 3 epochs.
7. Evaluate the model and generate the visualizations.

### Required Libraries

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

## Conclusion

This project demonstrates the use of ResNet-18 transfer learning for six-class scene image classification. Feature extraction achieved 92.55% validation accuracy, while fine-tuning increased it to 93.30%.

The results highlight the value of pretrained CNNs, fine-tuning, and comprehensive evaluation in developing image classification models.

## Author

**Anderia Nisar**

Bachelor's in Electronics

## Acknowledgment

Completed as part of a deep learning lab focused on image classification and transfer learning.
