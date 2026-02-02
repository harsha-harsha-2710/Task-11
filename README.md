# Task-11
# SVM – Breast Cancer Classification

## Project Overview
This project implements **Support Vector Machine (SVM)** models to classify breast cancer tumors as **malignant** or **benign** using the **Sklearn Breast Cancer dataset**.  
The task focuses on kernel-based classification, feature scaling, hyperparameter tuning, and model evaluation.

---

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numerical features
- **Target Classes:**
  - `0` → Malignant
  - `1` → Benign

---

## Tools & Libraries
- Python  
- Scikit-learn  
- Matplotlib  
- Joblib  

---

## Project Workflow
1. Loaded and explored the breast cancer dataset  
2. Applied **StandardScaler** to normalize feature values  
3. Split data into training and testing sets  
4. Trained a **baseline SVM with linear kernel**  
5. Trained an **SVM with RBF kernel** and compared performance  
6. Used **GridSearchCV** to tune hyperparameters (`C`, `gamma`)  
7. Evaluated the best model using:
   - Confusion Matrix  
   - Classification Report  
8. Plotted **ROC Curve** and calculated **AUC score**  
9. Saved the final trained **pipeline (Scaler + SVM)** for reuse  

---

## Model Evaluation
- **Metrics Used:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- **Visualization:**
  - ROC Curve plotted using Matplotlib

The tuned **RBF kernel SVM** achieved better performance than the linear kernel model.

---

##Saved Model
The final optimized model was saved as a pipeline containing:
- `StandardScaler`
- `Support Vector Classifier (SVC)`

**Saved file:**  
