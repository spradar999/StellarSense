# 🌌 StellarSense: Stellar Object Classification using Machine Learning

## Problem Statement

The objective of this project is to design an automated classification system using machine learning techniques to accurately classify celestial objects using photometric and spectroscopic features from the SDSS DR17 dataset. Multiple classification models are trained, evaluated, and compared to identify the most effective approach.

## Dataset Description

Dataset Name: Stellar Classification Dataset – SDSS17
Source: Sloan Digital Sky Survey (SDSS) Data Release 17
Task: Multi-class classification (Star, Galaxy, Quasar)

The dataset consists of astronomical observations of celestial objects captured through photometric and spectroscopic measurements. Each record corresponds to a single observed object and contains spatial coordinates, brightness values in multiple wavelength bands, and spectroscopic information.

## Citation

Fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. Retrieved February 10, 2026, from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17

## Acknowledgements

The data released by the Sloan Digital Sky Survey (SDSS) is in the public domain. This dataset is taken from the current data release DR17.
Abdurro’uf et al., The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar and APOGEE-2 Data (submitted to ApJS) arXiv:2112.02026

## 🔹 Key Attributes

alpha (Right Ascension) and delta (Declination): Sky coordinates

u, g, r, i, z: Photometric magnitudes in five wavelength bands

redshift: Wavelength shift indicating distance and velocity

run_ID, rerun_ID, cam_col, field_ID, plate, MJD, fiber_ID: Instrumental identifiers

class: Target label (STAR, GALAXY, QSO)

## 🔹 Dataset Characteristics

Type: Structured tabular data

Number of features: More than 12

Number of instances: Several thousand records

Target variable: class

Nature of problem: Multi-class classification

## Models Used

The following machine learning models were implemented:

Logistic Regression

Decision Tree

k-Nearest Neighbors (kNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)


## 📈 Performance Comparison

| Model               | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.957    | 0.988 | 0.955     | 0.948  | 0.951 | 0.925 |
| Decision Tree       | 0.966    | 0.970 | 0.960     | 0.961  | 0.960 | 0.939 |
| kNN                 | 0.903    | 0.952 | 0.920     | 0.864  | 0.889 | 0.826 |
| Naive Bayes         | 0.691    | 0.845 | 0.607     | 0.595  | 0.524 | 0.437 |
| Random Forest       | 0.979    | 0.995 | 0.978     | 0.973  | 0.976 | 0.963 |
| XGBoost             | 0.978    | 0.996 | 0.977     | 0.972  | 0.974 | 0.960 |

## 📝 Observation about Model Performance

| Model                     | Observation |
|---------------------------|-------------|
| Logistic Regression       | Shows strong performance with balanced precision, recall, and F1-score. High AUC indicates good class separability. Slightly weaker than ensemble models due to linear decision boundaries. |
| Decision Tree             | Captures non-linear patterns effectively and achieves high accuracy. However, it is more prone to overfitting compared to ensemble methods. |
| kNN                       | Moderate performance with good precision but lower recall. Sensitive to choice of k and computationally expensive for large datasets. |
| Naive Bayes               | Weakest performance due to unrealistic independence assumptions among astronomical features. |
| Random Forest (Ensemble)  | Best overall performer with high accuracy and stability. Ensemble learning reduces overfitting and improves generalization. |
| XGBoost (Ensemble)        | Comparable to Random Forest with excellent AUC and strong discrimination capability. Efficient at modeling complex relationships. |

