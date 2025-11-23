# Transfer Learning for Chest X-ray Classification

This project applies **Transfer Learning** to classify chest X-ray images, building on previous work. The goal is to develop a Python-based solution, experiment with different techniques, and justify the chosen model architecture.

## Project Objectives

- Implement **Transfer Learning** for chest X-ray classification.
- Experiment with **Data Augmentation**, **Feature Extraction**, and **Fine-Tuning**.
- Compare results and justify the best-performing model.

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python
  ```

## Experiments

### 1. Data Augmentation
- Apply transformations (rotation, zoom, flip) to increase dataset diversity.

### 2. Transfer Learning (TL)
- Use pre-trained models (e.g., VGG16, ResNet50, EfficientNet).

### 3. Feature Extraction
- Extract features from pre-trained models and train a classifier on top.

### 4. Fine-Tuning
- Unfreeze and retrain later layers of the pre-trained model.

## Results

- Compare accuracy, precision, recall, and F1-score across experiments.

## Justification

- Explain why the chosen architecture (e.g., ResNet50 + Fine-Tuning) outperforms others.
- Discuss trade-offs (e.g., training time vs. accuracy).
