# Heart Disease Prediction System

This project aims to predict the presence of heart disease in patients using machine learning algorithms. The system analyzes a dataset containing various health attributes and uses multiple classifiers to achieve the best accuracy. The project includes data analysis, preprocessing, model building, and evaluation.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Data Analysis](#data-analysis)
4. [Tech Stack](#tech-stack)
5. [Model Building](#model-building)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Usage](#usage)
9. [License](#license)

---

## Introduction
The goal of this project is to predict heart disease using a dataset containing 14 key attributes. The system employs various machine learning algorithms, including Logistic Regression, Naive Bayes, Random Forest, XGBoost, K-Nearest Neighbors, Decision Tree, and Support Vector Machine (SVM). The best-performing model is selected based on accuracy and other evaluation metrics.

---

## Dataset Overview
### Metadata
- **Source**: UCI Machine Learning Repository
- **Owner**: David Lapp
- **License**: Unknown (Public)
- **Databases**: Cleveland, Hungary, Switzerland, Long Beach V
- **Attributes**: 76 (14 used in this project)
- **Target Field**: Presence of heart disease (0 = no disease, 1 = disease)

### Attributes
1. Age
2. Sex (1 = male, 0 = female)
3. Chest pain type (4 values)
4. Resting blood pressure
5. Serum cholesterol (mg/dl)
6. Fasting blood sugar (>120 mg/dl)
7. Resting electrocardiographic results
8. Maximum heart rate achieved
9. Exercise-induced angina
10. ST depression induced by exercise
11. Slope of the peak exercise ST segment
12. Number of major vessels colored by fluoroscopy
13. Thalassemia (blood disorder)
14. Target (heart disease)

---

## Data Analysis
### Key Steps
1. **Exploratory Data Analysis (EDA)**:
   - Checked for missing values and outliers.
   - Analyzed distributions and correlations between variables.
   - Visualized data using Matplotlib and Seaborn.

2. **Outlier Removal**:
   - Identified and removed outliers in attributes like `trestbps`, `chol`, `ca`, and `oldpeak`.

3. **Feature Analysis**:
   - Analyzed the relationship between features and the target variable.
   - Visualized distributions of age, gender, chest pain, and other attributes.

---

## Tech Stack
- **Libraries**:
  - Pandas, NumPy (Data Manipulation)
  - Matplotlib, Seaborn (Visualization)
  - Scikit-learn (Model Building)
  - XGBoost (Boosting Algorithm)
  - MLxtend (Ensemble Learning)

---

## Model Building
### Algorithms Used
1. Logistic Regression
2. Naive Bayes
3. Random Forest
4. XGBoost
5. K-Nearest Neighbors (KNN)
6. Decision Tree
7. Support Vector Machine (SVM)

### Steps
1. **Data Splitting**:
   - Split the dataset into training and testing sets (80:20 ratio).

2. **Feature Scaling**:
   - Scaled the independent variables using `StandardScaler`.

3. **Model Training**:
   - Trained individual models and evaluated their performance using accuracy, confusion matrix, and classification report.

4. **Ensemble Learning**:
   - Used a stacking ensemble (StackingCVClassifier) with Decision Tree, Random Forest, and SVM as base models.

---

## Results
### Model Evaluation
| Model                     | Accuracy (%) |
|---------------------------|--------------|
| Logistic Regression        | 85.25        |
| Naive Bayes                | 80.33        |
| Random Forest              | 88.52        |
| XGBoost                    | 90.16        |
| K-Nearest Neighbors (KNN)  | 83.61        |
| Decision Tree              | 86.89        |
| Support Vector Machine (SVM)| **91.80**    |

### ROC Curve
- The ROC curve compares the performance of all models. SVM achieved the highest AUC score.

### Feature Importance
- The most important features for prediction are `ca`, `thal`, and `cp`.

---

## Conclusion
1. **Best Model**: Support Vector Machine (SVM) achieved the highest accuracy of **91.80%**.
2. **Key Insights**:
   - Exercise-induced angina and chest pain are major indicators of heart disease.
   - Males are more likely to have heart disease than females.
3. **Ensemble Learning**: Improved model accuracy by combining multiple classifiers.

---

## Usage
### Prerequisites
- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, MLxtend
