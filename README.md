# PCOS Classification Machine Learning Models

This repository implements and compares multiple supervised machine learning algorithms for predicting Polycystic Ovary Syndrome (PCOS) using clinical data. PCOS is a common hormonal disorder in women that can lead to irregular periods, acne, hair growth, weight gain, and infertility. PCOS is frequently misdiagnosed because its symptoms overlap with other conditions and can vary significantly between individuals.

Although the default dataset is PCOS clinical data, all models and pipelines are fully generalizable to any tabular dataset.

This repo includes:
- Standalone model training
- Stacking model training (multiple base models + meta-learner)
- Automated hyperparameter optimization (Bayesian Optimization)
- Model evaluation & comparison
- Customizable model testing
- Educational Jupyter notebooks for step-by-step learning

## What is StackingML?
Stacking is a method where a ML model learns how to best combine the results (predictions) provided by each base model. 
1. Each base model trains on the same training data
2. Their outputs are stacked into a new vector
3. The meta-learner model trains on this output vector and gives a final prediction

This method is effective because:
1. Different base models can capture different patterns
2. The meta-learner learns which model is more reliable for different cases

## Dataset:

This project uses the PCOS Diagnosis dataset from Kaggle: https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos

## Models:
Base Models
- Logistic Regression
- KNN
- Support Vector Machine
- Random Forest
- XGBoost
- AdaBoost
- Gaussian Naive Bayes
- Neural Network

Meta-Learners
- Logistic Regression
- Random Forest
- XGBoost
- AdaBoost
- Neural Network

You can construct stacking models using any combination of base models and a meta-learner

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Define dataset

To run the repo, you must download and place a dataset of your choosing (or the PCOS dataset) in your project directory, then create a .env file containing path to this dataset:
```bash
touch .env
```
## Running Models
You can train various ML or stacking ML models by customizing variables in testing.py
```bash
python3 testing.py
```
To iterate through all possible ML and stacking ML models defined in define_models.py, run compare.py, which will generate a .csv file as output.
```bash
python3 compare.py
```
All models will go through Bayesian hyperparameter tuning via BayesSearchCV to find the most optimal parameter combination.

Models are evaluated using: accuracy, precision, recall, F1 score, and confusion matrix

## Educational Notebooks
The BaseModelTesting folder contains code written in Jupyter notebooks with detailed explanation of each ML model for replication and learning purposes. 

Overview:

Artificial Intelligence (AI)
- Machine Learning (ML)
    - Supervised Learning → Regression, ✔️Classification
    - Unsupervised Learning → Clustering, Dimensionality Reduction, Association Rule Learning
    - Semi-Supervised Learning → Label Propagation
    - Reinforcement Learning → Q-Learning, Policy Gradient
- Deep Learning (DL)
    - ANN
    - CNN (images)
    - RNN / LSTM (sequences)
    - Transformers (text, vision)
    - Autoencoders
    - GANs
    - GNN
    - Deep Reinforcement Learning (DQN, A3C)
