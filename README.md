
# Vietnamese Toxic Speech Detection via Knowledge Distillation

This project explores **Knowledge Distillation (KD)** to compress a fine-tuned ViSoBERT teacher model into a lightweight TextCNN student model for Vietnamese toxic text classification.

The goal is to reduce model complexity and inference cost while maintaining competitive classification performance.

## Features

- Knowledge Distillation from ViSoBERT to TextCNN.
- Binary classification: Clean and Toxic.
- Hyperparameter search for distillation temperature and loss weight.
- Evaluation using Accuracy, Macro-F1, and Toxic F1-score.

## Repository Structure

```text
├── MODEL/
│   ├── TEACHER/ (on Hugging Face)
│   └── STUDENT/
├── TRAINING/
    ├── STUDENT_BASELINE.ipynb
    └── STUDENT_KD.ipynb
    └── VISOBERT_UIT.ipynb
├── COMPARISON.ipynb
├── GRID_SEARCH_VISUALIZATION.ipynb
├── .gitignore
└── README.md
```

## Pre-trained Model

The ViSoBERT teacher model is available on Hugging Face:

[Hungdz06/vietnamese-toxic-speech-nphungzz](https://huggingface.co/Hungdz06/vietnamese-toxic-speech-nphungzz)

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in the following order:

1. `STUDENT_BASELINE.ipynb` — Train and evaluate the baseline TextCNN.
2. `STUDENT_KD.ipynb` — Train TextCNN using Knowledge Distillation.
3. `GRID_SEARCH_VISUALIZATION.ipynb` — Compare hyperparameter configurations.

Place dataset directory.

## Model Architecture

- **Teacher:** ViSoBERT, fine-tuned for Vietnamese toxic text classification.
- **Student:** TextCNN with 300-dimensional embeddings, convolutional filter sizes of 2, 3, and 4, and 100 filters per size.
- **Distillation:** Combines supervised cross-entropy loss with teacher-student distillation loss.


## Experimental Results

The models were evaluated on the training, validation, and test sets using Accuracy, Macro-F1, and Toxic F1-score.

### Performance Comparison

| Dataset | Model | Accuracy | Macro-F1 | Toxic F1 |
|---|---|---:|---:|---:|
| Train | Teacher ViSoBERT | 99.54% | 99.53% | 99.46% |
| Train | Student TextCNN | 97.86% | 97.81% | 97.47% |
| Train | Student TextCNN-KD | 99.36% | 99.34% | 99.25% |
| Validation | Teacher ViSoBERT | 91.98% | 90.64% | 87.10% |
| Validation | Student TextCNN | 85.64% | 83.05% | 76.41% |
| Validation | Student TextCNN-KD | 87.49% | 85.17% | 79.29% |
| Test | Teacher ViSoBERT | 89.93% | 87.52% | 82.05% |
| Test | Student TextCNN | 86.24% | 82.69% | 74.85% |
| Test | Student TextCNN-KD | 87.33% | 83.80% | 76.24% |

### Key Findings

- **Knowledge Distillation improves student performance:** On the test set, TextCNN-KD improves Macro-F1 from 82.69% to 83.80% and Toxic F1 from 74.85% to 76.24% compared with the baseline TextCNN.
- **Competitive lightweight alternative:** TextCNN-KD achieves 87.33% test accuracy with approximately 2.72 million parameters.
- **Teacher remains the strongest model:** ViSoBERT achieves the highest test Macro-F1 (87.52%) and Toxic F1 (82.05%).
- **Generalization gap:** All models perform better on the training set than on the test set, indicating a gap between training and unseen-data performance.

Overall, knowledge distillation improves the TextCNN student's classification performance while retaining its compact architecture. However, the student model does not fully match the teacher's performance.


