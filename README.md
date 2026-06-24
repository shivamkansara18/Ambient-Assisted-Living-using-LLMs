# Early Anomaly Detection and Predictive Insights for Ambient Assisted Living (AAL) using LLMs

A multimodal Human Activity Recognition (HAR) system that combines wearable sensor data and visual data using a Cross-Attention Transformer architecture for Ambient Assisted Living (AAL) applications.

The system performs activity recognition, anomaly detection, and automated healthcare report generation using a locally deployed Mistral-7B Large Language Model.

---

## Overview

Traditional Human Activity Recognition systems rely on either:

- Wearable sensor data
- Camera/image-based recognition

Each modality has limitations.

Sensor-based systems lack environmental context, while vision-based systems suffer from occlusion, lighting conditions, and privacy concerns.

This project introduces a multimodal fusion framework that combines both modalities to improve robustness and accuracy for healthcare monitoring and elderly-care applications.

The system:

1. Extracts features from images and wearable sensor signals.
2. Aligns independent datasets using a novel pseudo-pairing strategy.
3. Fuses multimodal representations using a Cross-Attention Transformer.
4. Predicts human activities.
5. Generates structured clinical reports using Mistral-7B.

---

## Key Features

### Multimodal Activity Recognition

Combines:

- RGB Image Data
- Wearable Sensor Data

for improved activity recognition performance.

### Cross-Attention Fusion Transformer

Uses:

- Self-Attention
- Cross-Attention
- Shared Embedding Space

to learn relationships between sensor and visual modalities.

### Pseudo-Pairing Framework

Unlike traditional multimodal systems, this project does **not require synchronized datasets**.

A novel pseudo-pairing strategy:

- Pairs samples using semantic activity labels
- Eliminates dependency on synchronized recordings
- Enables multimodal training using independent datasets

### Clinical Report Generation

Predicted activities are converted into:

- Patient activity summaries
- Detected anomalies
- Caregiver recommendations

using a locally deployed **Mistral-7B** model.

### Automated PDF Reports

Generates healthcare-ready reports containing:

- Activity distributions
- Confidence scores
- Anomaly statistics
- LLM-generated summaries

---

## Architecture

```text
                ┌─────────────────┐
                │ RGB Images      │
                └────────┬────────┘
                         │
               Swin Transformer
                         │
                    1024-d Features
                         │
                         ▼

                ┌─────────────────┐
                │ Sensor Data     │
                └────────┬────────┘
                         │
                      MLP Encoder
                         │
                      64-d Features
                         │
                         ▼

            Shared Projection Space
                    (256-d)

                         │
                         ▼

          Cross-Attention Transformer
                         │
                         ▼

              Activity Classification
                         │
                         ▼

                LLM Report Generator
                    (Mistral-7B)
                         │
                         ▼

                 PDF Medical Report
```

---

## Datasets

### UCI-HAR Dataset

Sensor-based Human Activity Recognition dataset containing:

- Accelerometer Data
- Gyroscope Data

Activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Dataset:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

---

### Kaggle HAR Image Dataset

RGB image dataset containing human activity classes.

Only activities relevant to AAL were retained:

- Walking
- Running
- Sitting
- Standing
- Sleeping

Sports-specific activities were excluded.

---

## Methodology

### Phase 1 — Ontology Bridge

Creates a unified activity space across datasets.

Example:

| Image Activity | Unified Class |
|---------------|---------------|
| Walking | Walking |
| Running | Walking |
| Sleeping | Laying |

---

### Phase 2 — Feature Extraction

#### Image Encoder

Model:

- Swin Transformer Base

Output:

```text
1024-dimensional embeddings
```

#### Sensor Encoder

Model:

```text
MLP
512 → 256 → 128 → 64
```

Output:

```text
64-dimensional embeddings
```

---

### Phase 3 — Pseudo-Pairing

Creates multimodal training pairs using shared labels.

Example:

```text
Walking Image
        +
Walking Sensor Window

→ Pseudo Pair
```

This removes the requirement for synchronized recordings.

---

### Phase 4 — Cross-Attention Transformer

The fusion network performs:

- Self-attention within each modality
- Cross-attention between modalities
- Feed-forward refinement

The model learns complementary information from both streams.

---

### Phase 5 — Training

Regularization techniques:

- Dropout
- Weight Decay
- Label Smoothing
- Gradient Clipping
- Cosine Annealing Scheduler
- Modality Dropout

---

## Results

| Model | Validation Accuracy |
|---------|---------|
| Image Only | 87.2% |
| Sensor Only | 96.5% |
| Multimodal Fusion | **97.4%** |

The fusion model outperformed both unimodal baselines.

---

## LLM-Powered Clinical Reports

The activity predictions are summarized into structured healthcare reports.

Example:

```text
Patient Activity Summary

The patient spent most of the monitored session
performing walking and standing activities.

No significant inactivity periods were detected.

Overall mobility appears normal.

Recommendation:
Continue regular monitoring and encourage
daily walking activity.
```

---

## Tech Stack

### Machine Learning

- PyTorch
- Torchvision
- timm
- NumPy

### Computer Vision

- Swin Transformer

### Deep Learning

- Transformers
- Cross-Attention Networks

### LLM

- Mistral-7B
- MLX

### Reporting

- ReportLab

---

## Project Structure

```text
project/

├── phase1_ontology.py
├── phase2_encoders.py
├── phase3_dataset.py
├── phase4_model.py
├── phase5_train.py
│
├── llm_report_generator.py
│
├── data/
├── checkpoints/
├── reports/
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/AAL-Multimodal-HAR.git

cd AAL-Multimodal-HAR

pip install -r requirements.txt
```

---

## Training

```bash
python phase5_train.py
```

---

## Generate Clinical Reports

```bash
python llm_report_generator.py
```

---

## Future Work

- Real-time HAR deployment
- Edge-device inference
- Multi-patient monitoring
- Federated learning
- Fall detection
- Domain adaptation for elderly populations
- Retrieval-Augmented Generation (RAG)

---

## Authors

- Shivam Kansara

Department of Computer Science & Engineering

Pandit Deendayal Energy University

---

## Citation

If you use this work in your research, please cite:

```bibtex
@project{aal_har_2026,
  title={Early Anomaly Detection and Predictive Insights for Ambient Assisted Living using LLMs},
  author={Kansara, Shivam and Surti, Khushi and Shah, Pal and Rathwa, Jaydeep},
  year={2026}
}
```

---

## License

MIT License
