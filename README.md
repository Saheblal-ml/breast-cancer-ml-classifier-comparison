# Machine Learning Assignment 2 — Classification Model Comparison

This project compares five supervised machine learning classifiers using a common dataset and six evaluation metrics. I trained and evaluated the models, tested the Streamlit application with held-out labeled data, and reproduced the execution in the BITS Pilani Virtual Lab.

## a. Problem statement
Compare five supervised classification models on a common dataset using six evaluation metrics, then deploy an interactive application to evaluate saved models on uploaded labeled test data **without retraining**.

## b. Dataset description
**Source:** UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`.

The dataset contains **569 records**, **30 numerical features**, and a binary target. The target is recoded so **1 = malignant** and **0 = benign**. All five classifiers use the same stratified 80/20 train/test split (`random_state=42`). The data and application are intended for educational ML benchmarking, **not medical decision-making**.

## c. GitHub repository
https://github.com/Saheblal-ml/breast-cancer-ml-classifier-comparison

## d. Models used and evaluation
1. Logistic Regression (with StandardScaler)
2. Decision Tree Classifier (`max_depth=5`, `min_samples_leaf=3`)
3. K-Nearest Neighbors (`k=7`, with StandardScaler)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, with StandardScaler)

**Evaluation metrics:** Accuracy, ROC AUC (using predicted probabilities), Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC). The positive class for binary classification metrics is **malignant (1)**. Saved model pipelines include preprocessing steps where applicable.

### Model comparison table
The following results were reproduced by running `python train_models.py` in the BITS Pilani Virtual Lab and verified in the Streamlit application using the held-out test CSV.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations about model performance
These observations are based on the measured results for the common held-out test set.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved strong accuracy (96.49%) and the highest AUC (99.60%) among the five models. |
| Decision Tree | Had the lowest accuracy (90.35%) and the lowest malignant-class recall (78.57%), indicating more missed malignant cases than the other models. |
| KNN | Achieved good accuracy (95.61%), but performed slightly below Logistic Regression and SVM on accuracy and F1-score. |
| Naive Bayes | Achieved perfect precision (100%) for malignant predictions, but its recall was lower (83.33%), meaning some malignant cases were missed. |
| Support Vector Machine | Achieved the highest accuracy (97.37%), F1-score (96.30%), and MCC (94.42%). Its malignant-class recall was 92.86%. |

**Overall winner — Support Vector Machine (SVM):** SVM performed best overall on this test set because it achieved the highest accuracy, F1-score, and MCC. Its precision was 100%, and the confusion matrix showed **72 correctly identified benign cases, 0 benign cases incorrectly flagged as malignant, 39 correctly identified malignant cases, and 3 malignant cases incorrectly predicted as benign**. Thus, despite its strong overall results, it missed **3 of the 42 malignant cases**. Logistic Regression had the highest AUC (0.9960), slightly exceeding SVM's AUC (0.9947). These findings describe performance on this particular held-out test set, not guaranteed performance on new data.

## Run and reproduce

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload the generated `test_data.csv` in the Streamlit application. Switch between the five saved models to view six metrics, a confusion matrix, and a classification report. The included test CSV comes from the **held-out split** and was not used for fitting. The application loads the saved models and does **not retrain** them when evaluating an uploaded file.

## Streamlit application
**Live application:** https://breast-cancer-ml-classifier-comparison-2k7mtlqbtsbbl6jtuaay9p.streamlit.app/

The application allows a user to upload a labeled test CSV, select one of the five classifiers, and inspect evaluation metrics, the confusion matrix, and the classification report.

## BITS Virtual Lab execution
I cloned the GitHub repository into the BITS Pilani Prayogshala Virtual Lab, installed the project dependencies, executed `python train_models.py`, and ran the Streamlit application in the Virtual Lab browser. I uploaded `test_data.csv` and checked the results for all five models. The execution screenshot is included separately in the assignment submission PDF.

## Academic integrity
This repository documents an educational machine learning exercise. The results and interpretations should be reviewed and understood before submission, in accordance with BITS Pilani academic-integrity requirements.

before submission, in accordance with BITS Pilani academic-integrity requirements.

