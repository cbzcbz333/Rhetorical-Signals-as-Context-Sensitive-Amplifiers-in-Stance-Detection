# Rhetoric-Aware Stance Detection

This repository contains the code for our study on **rhetoric-aware stance detection**, which models rhetorical strategies as explicit and interpretable features rather than treating them as latent sentiment signals.  
The goal is to analyze **how stance is expressed** through discourse mechanisms, and to examine their **topic-dependent effects** under controlled experimental settings.

## Overview

We propose a modular framework that integrates rhetorical features with standard lexical and neural models for stance detection.  
The rhetorical categories considered include:

- Rhetorical questions  
- Epistemic modality and hedging  
- Contrast and opposition structures  

Experiments are conducted on the **SemEval-2016 Task 6** dataset, using both feature-based (TF–IDF + Logistic Regression) and neural (e.g., BERT-based) models, with emphasis on topic-wise evaluation and diagnostic analysis.

## Repository Structure

.
├── data/ # Data preparation scripts (SemEval-2016)

├── features/ # Rhetorical feature extraction modules

├── models/ # Linear and neural stance detection models

├── scripts/ # Training and evaluation scripts

├── configs/ # Model and experiment configurations

├── analysis/ # Feature weight analysis and error inspection

├── requirements.txt

└── README.md


## Data

This work uses the publicly available **SemEval-2016 Task 6** stance detection dataset.  
Due to licensing constraints, the dataset is not redistributed in this repository.

Please obtain the data from the official SemEval source and place it in the `data/` directory following the instructions in `data/README.md`.

## Installation

We recommend using Python 3.8+.

bash
pip install -r requirements.txt

## Running Experiments
Feature-based model (TF–IDF + Logistic Regression)
python scripts/train_lr.py --config configs/lr.yaml
Neural model (e.g., BERT baseline and rhetoric-augmented model)
python scripts/train_bert.py --config configs/bert.yaml
All experiments use fixed random seeds for reproducibility.

## Reproducibility

The repository provides:

Fixed train/dev/test splits

Explicit feature extraction pipelines

Configuration files for all reported experiments

Using the provided scripts and configurations, the results reported in the paper can be reproduced.

## License

This project is released under the MIT License.

## Citation

If you use this code, please cite the corresponding paper:

@article{anonymous2025rhetoric,
  title={Rhetoric-Aware Stance Detection with Interpretable Discourse Features},
  journal={Information Processing \& Management},
  year={2025}
}
