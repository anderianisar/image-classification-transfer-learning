# Image Classification with Transfer Learning

**Project 2: Deep Learning Applications** | Introduction to Applied AI | Department of Electronics  
Individual project by **Anderia Nisar**

This repository contains my Project 2 work: a CNN-based image classifier built with **PyTorch** and **transfer learning** on a pretrained **ResNet-18**. Week 5 builds the full pipeline on a benchmark scene dataset; Week 6 applies the same pipeline to a medical imaging problem (malaria detection).

---

## Contents

| File | Description |
|---|---|
| `Week5_Transfer_Learning_Complete.ipynb` | Full Kaggle notebook: code, outputs, and written analysis for every part |
| `resnet18_intel.pt` | Saved weights of the final model (44.8 MB) |
| `images/` | Figures used in this README (learning curves, confusion matrix, error gallery) |

---

## Week 5: Scene Classification with Transfer Learning

### Overview
A ResNet-18 pretrained on ImageNet (1.2 million images) is adapted to recognise **6 scene classes**: buildings, forest, glacier, mountain, sea, and street. The work follows a staged approach: understand the data and a convolution by hand, build clean data pipelines, train only a new classifier head (feature extraction), then fine-tune the last residual block, and finally evaluate once on the test set and study the errors.

### Dataset
**Intel Image Classification** (Kaggle, 6 classes, 150 × 150 RGB images)

| Split | Images | Notes |
|---|---|---|
| Train | 11,929 | 85% of the original training folder, **augmented** (random resized crop, horizontal flip, mild colour jitter) |
| Validation | 2,105 | 15% of the original training folder, clean (resize + normalise only) |
| Test | 3,000 | Original `seg_test` folder, clean, **used only once** |

All images are resized to 224 × 224 and normalised with the ImageNet mean and standard deviation, which the pretrained weights expect.

### Method

| Stage | What was trained | Epochs | Learning rate | Best validation accuracy |
|---|---|---|---|---|
| 1. Feature extraction | New 6-class head only (**3,078** parameters, 0.028% of 11.2M) | 3 | 1e-3 | **0.901** |
| 2. Fine-tuning | Head + `layer4` (**8,396,806** parameters) | 3 | 1e-4 (`layer4`), 1e-3 (head) | **0.931** |

Training uses Adam, cross-entropy loss, batch size 64, and keeps the weights of the best validation epoch. Using a smaller learning rate for the pretrained block than for the new head adapts the good pretrained weights gently instead of destroying them.

### Results

| Metric | Value |
|---|---|
| Feature extraction, validation accuracy | 0.901 |
| Fine-tuning, validation accuracy | 0.931 (+3.04 points) |
| **Test accuracy** | **0.936** (2,809 of 3,000 correct) |
| Macro-average F1 | 0.937 |

**Per-class test results**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| buildings | 0.954 | 0.908 | 0.931 |
| forest | 0.996 | 0.987 | 0.992 |
| glacier | 0.920 | 0.870 | 0.894 |
| mountain | 0.883 | 0.920 | 0.901 |
| sea | 0.947 | 0.980 | 0.963 |
| street | 0.930 | 0.958 | 0.944 |

**Learning curves** (dashed line = start of fine-tuning)

![Learning curves](images/learning_curves.png)

Both losses fall during feature extraction. Fine-tuning gives a further drop, and at epoch 6 validation loss starts to rise while training loss keeps falling, an early sign of overfitting. The best validation epoch is therefore kept.

**Confusion matrix (test set)**

![Confusion matrix](images/confusion_matrix.png)

### Error Analysis
- The largest confusion is **glacier ↔ mountain** (57 glaciers predicted as mountain and 31 mountains predicted as glacier, 88 of the 191 test errors). Snow-covered rocky peaks appear in both classes.
- The second largest is **buildings ↔ street** (34 + 17 = 51 errors).
- These two pairs account for about 73% of all errors.
- Most inspected errors are **not clear model failures**: many photos labelled "buildings" show shop fronts with cars and a road, and the model predicted "street" with up to 0.99 confidence. These images are genuinely ambiguous or arguably mislabelled, so remaining errors at this accuracy level are mostly a label-quality issue rather than a model-capacity issue.

**Error gallery** (first eight test errors, with true label, predicted label, and confidence)

![Error gallery](images/error_gallery.png)

### Key Takeaways
- Training only **3,078 of 11.2 million parameters** already gives about 90% accuracy, because the pretrained layers provide general features (edges, textures, parts).
- Fine-tuning the last block with a smaller learning rate adds about 3 accuracy points.
- Validation is used for all choices (best epoch, comparing strategies); the test set is touched once.
- Augmentation is applied to the training set only; evaluation always uses the plain transform.

---

## Week 6 Proposal: Malaria Detection from Blood-Cell Images

**Problem.** Malaria is diagnosed in the field mainly by examining stained blood smears under a microscope, which is slow and depends on a trained expert, and a missed infection can have serious consequences for the patient. An image classifier that screens single blood-cell images as **parasitized** or **uninfected** could support laboratory technicians and health workers by flagging suspicious cells quickly and consistently, especially where trained microscopists are scarce.

**Dataset.** I will use the Kaggle "Cell Images for Detecting Malaria" dataset, which comes from the US National Institutes of Health: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria. It contains **27,558 segmented cell images in 2 classes** (13,779 parasitized and 13,779 uninfected), taken from thin blood-smear slides. The data is balanced, and the image sizes are not uniform, so all images will be resized to 224 × 224. The data is released by the NIH as public-domain data; I will still confirm the exact license terms on the Kaggle page before use.

**Approach.** I will reuse the Week 5 pipeline: ImageNet-normalised inputs, a train / validation / test split with augmentation on the training set only, ResNet-18 feature extraction followed by fine-tuning of `layer4` with two learning rates, and a single final evaluation on a held-out test set. Because this is a medical screening task, I will report **recall (sensitivity) for the parasitized class** and the number of false negatives in addition to accuracy, and I will study the misclassified cells. Augmentations will be chosen to suit microscope images (flips and rotations are safe because cells have no fixed orientation, unlike the scene photos in Week 5).

**Limitations to discuss.** The images are pre-segmented single cells from a curated set, so real full-slide smears with artefacts and noise may be harder, and results on this dataset should not be taken as clinical performance.

---

## How to Reproduce
1. Create a Kaggle notebook, add the **Intel Image Classification** dataset (`puneet6060/intel-image-classification`), and set **Accelerator = GPU T4** and **Internet = On**.
2. Run `Week5_Transfer_Learning_Complete.ipynb` from top to bottom (about 40 s per epoch on a T4).
3. To reload the saved model:
   ```python
   import torch, torch.nn as nn
   from torchvision import models
   model = models.resnet18()
   model.fc = nn.Linear(512, 6)          # recreate the 6-class head first
   model.load_state_dict(torch.load('resnet18_intel.pt', map_location='cpu'))
   model.eval()
   ```

## Tech Stack
Python, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Kaggle GPU (Tesla T4).

## Repository
https://github.com/anderianisar/image-classification-transfer-learning




# Week-6  Malaria Detection from Blood-Cell Images

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

# Week 7: Text and Multimodal AI
### Malaria diagnosis support: blood-smear image + clinical text case review

![Python](https://img.shields.io/badge/Python-3.13-blue)
![spaCy](https://img.shields.io/badge/spaCy-3.8-09a3d5)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-red)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20beff)

A coursework project for **Introduction to Applied AI**. It builds a small text-processing pipeline and connects it to the **Week 6 blood-cell classifier** (`Parasitized` vs `Uninfected`), producing a privacy-safe **case record** that combines image evidence and clinical-note evidence for a human reviewer.

> **Disclaimer.** This is an educational project. It is decision support only and is **not** a medical diagnosis tool. A qualified human makes every clinical decision.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [What the Notebook Covers](#what-the-notebook-covers)
- [Key Results](#key-results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Data and Model Sources](#data-and-model-sources)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Privacy and Ethics](#privacy-and-ethics)
- [Author](#author)

---

## Overview

Each part of the notebook is **implemented, measured, then interpreted** using the output of the cell directly above the explanation. The emphasis is on honest evaluation: held-out data, negative tests and reading errors one by one, not only reporting averages.

## Pipeline

```
 blood-smear image ──► ResNet-18 (Week 6) ───────┐
                                                 ├──► case record ──► REFER / ROUTINE ──► human review
 clinical note ──► mask PII ──► spaCy + regex ───┘
```

## What the Notebook Covers

| # | Part | Topic | Method |
|---|---|---|---|
| 1 | TF-IDF by hand | Rebuild one vectoriser row manually | NumPy |
| 2 | Text classification | Spam detection with word vs character n-grams | TF-IDF + Logistic Regression |
| 3 | Named Entity Recognition | Out-of-the-box vs rule-augmented, with a held-out set | spaCy + EntityRuler |
| 4 | Regex fields | Fixed-format extraction with negative tests | `re` |
| 5 | CER and OCR repair | Measuring OCR quality, fixing look-alike characters | Levenshtein, EasyOCR |
| 6 | Image + text case record | Privacy-first record combining both evidence sources | ResNet-18 + spaCy + regex |
| 7 | Check Your Understanding | Written answers with working | |

## Key Results

| Part | What was measured | Result |
|---|---|---|
| 1. TF-IDF | Hand reproduction of one row | 4 of 4 values match scikit-learn; cosine(d2, d3) = 0.544 |
| 2. Spam | Macro-F1 on 1,115 test messages | Word 1-2 grams **0.974**, Character 2-5 grams **0.982** |
| 3. NER | Provided 6-sentence set, F1 | 0.387 to **0.688** with entity rules |
| 3. NER | Own domain, rule-writing set | 0.529 to 0.919 (optimistic) |
| 3. NER | Own domain, **held-out** set | 0.556 to **0.778** (honest estimate) |
| 4. Regex | 12 test cases (5 negative of my own) | **12 / 12** behaved as expected |
| 5. OCR | Controlled error pairs, CER | 0.062, 0.062, 0.143 to **0.000** after repair |
| 5. OCR | EasyOCR on a clean synthetic invoice | CER **0.000** |
| 6. Case record | Parasitized image + fever note | **REFER**; phone and CNIC leak checks both **False** |
| 6. Case record | Healthy image + urgent note | Still **REFER** (the note is the only reason) |

### Spam classifier comparison

| Metric (test set) | Word 1-2 grams | Char 2-5 grams |
|---|---|---|
| Features | 12,128 | 36,449 |
| Macro-F1 | 0.974 | **0.982** |
| Spam precision | 0.979 | **0.986** |
| Spam recall | 0.933 | **0.953** |
| Accuracy | 0.988 | **0.992** |

## Key Findings

1. **Accuracy can mislead.** With 13.4 % spam, an "always ham" model scores 86.6 %, so macro-F1 and per-class recall are used instead.
2. **Models learn shortcuts.** Features such as `uk`, `to` and `your` describe the dataset, not spam. An ordinary Roman Urdu message was flagged as spam because it contained `call`, `free` and `mobile`.
3. **Evaluate on held-out data.** The same NER rules gave +0.390 F1 on the sentences they were written from but only +0.222 on unseen ones.
4. **Choose the tool for the error.** Regex for fixed formats, rules for closed boundary problems, trained models for open-ended names.
5. **Negative tests matter.** Each deliberately sloppy regex passed positive examples and failed on text designed to attack it.
6. **One OCR character can break a field.** `3O/09/2O26` is only 14 % wrong by CER but would not match the date regex, so repair before extraction.
7. **Mask first, extract second.** Identifiers never reach the record, verified by searching the final JSON.
8. **Fail towards referral.** Any single source of evidence is enough to refer a case, and disagreements stay visible to a human.

## Tech Stack

| Area | Libraries |
|---|---|
| Data | `numpy`, `pandas`, `matplotlib` |
| Classical ML | `scikit-learn` (TF-IDF, Logistic Regression, metrics) |
| NLP | `spaCy` 3.8 with `en_core_web_sm` |
| OCR | `easyocr`, `Pillow` |
| Deep learning | `torch`, `torchvision` (ResNet-18) |

## Data and Model Sources

| Resource | Used in | Notes |
|---|---|---|
| [SMS Spam Collection (UCI)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Part 2 | 5,572 messages: 4,825 ham, 747 spam |
| [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) | Part 6 | Folders `Parasitized` and `Uninfected` |
| Week 6 checkpoint (ResNet-18, 224 x 224, RGB) | Part 6 | Own model from the previous lab; not included in this repository |

The datasets and the model checkpoint are **not** stored in this repository. Add them as inputs in your Kaggle notebook, or download them separately.

## Getting Started

### Option 1: Kaggle (recommended)

1. Upload `week-7-text-and-multimodal-AI-report.ipynb` to a new Kaggle notebook.
2. Add the two datasets and your Week 6 checkpoint as **Inputs**.
3. Turn **Internet On** in the notebook settings (EasyOCR downloads its model on first use).
4. Run all cells in order.

### Option 2: Local machine

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib scikit-learn spacy pillow easyocr torch torchvision jupyter
python -m spacy download en_core_web_sm

jupyter notebook week-7-text-and-multimodal-AI-report.ipynb
```

> The notebook uses Kaggle paths such as `/kaggle/input/...`. When running locally, change those paths to where you stored the datasets and the checkpoint.

## Project Structure

```
.
├── week-7-text-and-multimodal-AI-report.ipynb   # full report: code, outputs, analysis
└── README.md
```

## Limitations

| Area | Limitation |
|---|---|
| Spam | One random split; English and UK-heavy data; some labels are wrong |
| NER | Only 10 domain sentences (36 entities); one entity moves F1 by about 0.056 |
| OCR | Tested on a clean synthetic image; real-scan CER is an expectation, not a result |
| Image model | Trained on single-cell images from one public dataset; needs validation on local laboratory images |
| Case record | Raw softmax confidence (not calibrated); no explicit conflict flag; entity list partly unreliable |

### Possible next steps

- Add a `conflict` field and calibrated probabilities to the case record.
- Annotate a few hundred domain sentences and train a small NER model for names.
- Measure CER on real, hand-transcribed laboratory reports.
- Test a lower spam threshold on the full test set, not only on the errors.

## Privacy and Ethics

- All names, places, phone numbers, CNICs and amounts in the examples are **synthetic**.
- Personal identifiers (CNIC, mobile, email) are masked **before** any entity or field extraction.
- The output is **decision support, not a diagnosis**, and the final decision always stays with a qualified clinician.

## Author

**[Anderia Nisar]**
Introduction to Applied AI, Week 7
