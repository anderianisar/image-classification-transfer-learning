
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



# Malaria Detection from Blood-Cell Images

Transfer learning with **ResNet-18** to classify single blood cells as **Parasitized** or **Uninfected**, evaluated with a **patient-level split** to avoid data leakage.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-orange)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20(GPU)-20BEFF)
![Task](https://img.shields.io/badge/Task-Binary%20Image%20Classification-green)

> **Course:** Introduction to Applied AI | **Project 2:** Deep Learning Applications | **Week 6**
> **Author:** [Anderia Nisar]

---

## Table of Contents

1. [Overview](#overview)
2. [Results at a Glance](#results-at-a-glance)
3. [Dataset](#dataset)
4. [Method](#method)
5. [Detailed Results](#detailed-results)
6. [Explainability (Grad-CAM)](#explainability-grad-cam)
7. [Repository Contents](#repository-contents)

---

## Overview

Malaria is diagnosed in the field by examining blood smears under a microscope, which is slow and depends on the skill of the person counting. This project trains a deep-learning classifier that looks at a single segmented red blood cell and decides whether it contains a malaria parasite.

The focus is not only on accuracy but on **trustworthy evaluation**:

- **Patient-level split:** the 27,558 images come from only 200 patients (about 138 cells each), so a random split would leak patient and slide information into the test set. All splits are grouped by patient id.
- **Leakage audit:** exact duplicates (MD5) and near-duplicates (image fingerprints) were checked before training.
- **Recall-oriented threshold:** the decision threshold was chosen on validation data to reach at least 95% recall for the Parasitized class, then applied once to the test set.
- **Prevalence-aware reporting:** PPV is reported at the test prevalence and recomputed for 5% and 1% prevalence.
- **Shortcut check:** Grad-CAM was used to test whether the model looks at the cell or at the black background.

## Results at a Glance

| Metric (test set, 3,335 cells from 28 unseen patients) | Value |
|---|---|
| Accuracy (threshold 0.5) | 0.970 |
| Macro-F1 (threshold 0.5) | 0.969 |
| AUC | 0.994 |
| Sensitivity at threshold 0.160 | 0.982 |
| Specificity at threshold 0.160 | 0.941 |
| PPV at test prevalence (40.2%) | 0.918 |
| PPV at 5% prevalence | 0.468 |
| PPV at 1% prevalence | 0.145 |

## Dataset

**Cell Images for Detecting Malaria** (NIH, hosted on Kaggle)

- Kaggle: <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>
- Original source: <https://ceb.nlm.nih.gov/repositories/malaria-datasets/>

| Property | Value |
|---|---|
| Total images | 27,558 |
| Classes | Parasitized (13,779) and Uninfected (13,779) |
| Colour mode | RGB |
| Image size | width 61 to 229 px, height 82 to 187 px (median about 130 x 133) |
| Patients | 200 (id taken from the file name, e.g. `C100P61ThinF_IMG_...` gives `C100`) |
| Patient class mix | 150 patients with both classes, 50 with Uninfected only, 0 with Parasitized only |

Expected folder layout (one sub-folder per class):

```
cell_images/
├── Parasitized/
└── Uninfected/
```

### Data splits

| Split | Images | Patients | Parasitized | Uninfected |
|---|---|---|---|---|
| Train | 19,118 | 139 | 9,586 | 9,532 |
| Validation | 5,105 | 33 | 2,853 | 2,252 |
| Test | 3,335 | 28 | 1,340 | 1,995 |

No patient appears in more than one split.

## Method

```
Index files + patient ids
        ↓
Audit: exact duplicates (MD5) and near-duplicates (32x32 fingerprints)
        ↓
Patient-level split (StratifiedGroupKFold): train / validation / test
        ↓
Augmentation + ImageNet normalization, images preloaded in RAM
        ↓
Phase 1: train the new head only
        ↓
Phase 2: fine-tune layer4 + head
        ↓
Test evaluation, threshold chosen on validation, PPV vs prevalence
        ↓
Grad-CAM shortcut check, then export of the model
```

### Model

ResNet-18 pretrained on ImageNet with a new 2-class linear head (11,177,538 parameters in total).

| Phase | Trainable layers | Trainable parameters | Epochs | Learning rate | Best validation macro-F1 |
|---|---|---|---|---|---|
| 1 | Head only | 1,026 (0.009%) | 2 | 1e-3 | 0.866 |
| 2 | `layer4` + head | 8,394,754 (75.1%) | 3 | 1e-4 (layer4), 1e-3 (head) | 0.937 |

The best epoch of each phase is selected by **validation macro-F1**. The test set is used only once, at the end.

### Training setup

| Setting | Value |
|---|---|
| Input size | 224 x 224 |
| Optimizer | Adam |
| Batch size | 32 (training), 64 (evaluation) |
| Loss | Cross-entropy with class weights 0.997 (Parasitized) and 1.003 (Uninfected) |
| Augmentation | random resized crop (scale 0.8 to 1.0), rotation up to 10 degrees, horizontal and vertical flip, brightness and contrast jitter of 0.15 |
| Normalization | ImageNet mean and standard deviation |
| Random seed | 42 |

Hue and saturation jitter is not used, because the purple stain of the parasite may be real evidence.

## Detailed Results

### Default threshold (0.5)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Parasitized | 0.963 | 0.962 | 0.963 | 1,340 |
| Uninfected | 0.974 | 0.975 | 0.975 | 1,995 |

| Confusion matrix | Predicted Parasitized | Predicted Uninfected |
|---|---|---|
| **True Parasitized** | 1,289 | 51 |
| **True Uninfected** | 49 | 1,946 |

### Recall-oriented threshold (0.160)

The threshold was chosen on the validation set for recall of at least 0.95 (validation recall 0.950, specificity 0.936).

| Threshold | Missed infections (FN) | False alarms (FP) |
|---|---|---|
| 0.5 (default) | 51 | 49 |
| **0.160** | **24** | **117** |

Test result at 0.160: TP 1,316, FN 24, TN 1,878, FP 117, giving sensitivity 0.982, specificity 0.941 and PPV 0.918.

### PPV depends on prevalence

PPV = (sens x prev) / (sens x prev + (1 - spec) x (1 - prev))

| Prevalence | PPV |
|---|---|
| 40.2% (test set) | 0.918 |
| 5% | 0.468 |
| 1% | 0.145 |

Sensitivity and specificity do not change with prevalence, but PPV falls sharply when infected cells are rare. In a real screening setting, more than half of the positive calls could be false alarms, so a positive result needs confirmation by a microscopist.

## Explainability (Grad-CAM)

Grad-CAM on the last convolutional block (`layer4`) was used to check for shortcuts, since stain colour and black padding vary from slide to slide.

| Class | Background share of pixels | Background share of attention |
|---|---|---|
| Parasitized | 0.272 | 0.121 |
| Uninfected | 0.262 | 0.163 |
| Overall | 0.266 | 0.145 |

**Findings**

- The background gets less attention than its share of the pixels in both classes, which argues against a strong background shortcut.
- Correct Parasitized predictions can focus tightly on a dark parasite spot.
- Some maps peak near the cell edge, and the model made confident errors on odd-shaped cells, so the cell outline may act as a clue.

**Verdict:** shortcut suspected (mild), not confirmed. A suggested follow-up is to paint the padding and the outer ring of each test cell with a fixed colour and re-score at the same threshold; a sharp drop in sensitivity or specificity would confirm reliance on the outline.

## Repository Contents

| File | Description |
|---|---|
| `week-6-malaria-detection-professional.ipynb` | Full notebook with code, outputs and written analysis |
Saved model (44.8 MB), created by the notebook in `/kaggle/working
Backup of the image index with file sizes and MD5 hashes, created by the notebook |
| `README.md` | This file |

The checkpoint `week6_model.pt` contains five items: `state_dict`, `classes`, `backbone`, `img_size` and `grayscale`. The decision threshold **0.160 is not stored** in the file and must be applied separately.
## Limitations

- **Not a diagnostic tool.** Positive results must be confirmed by a trained microscopist.
- **Curated data.** The images are already segmented single cells on black padding. Real blood smears contain artefacts, overlapping cells and noise that are not present here.
- **Small test set of patients.** The test set has 3,335 cells but only 28 patients, and the grouped split cannot force a balanced class mix (40.2% Parasitized in test against 55.9% in validation).
- **Single run.** Results come from one split and one random seed, so no confidence intervals are reported.
- **Possible shortcut.** The Grad-CAM check was done on six heatmaps and 300 attention scores, so the mild shortcut suspicion is neither confirmed nor ruled out.
- **Prevalence.** PPV depends strongly on how common infected cells are; the reported 0.918 holds only at 40.2% prevalence.
