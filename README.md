# Machine Learning Assignment 2 — Classification Model Comparison

> Educational starter project. Review, customize, run, explain, and validate the work yourself before submission. Follow BITS Pilani academic-integrity rules.

## a. Problem statement
Compare five supervised classification models on a common dataset using six evaluation metrics, then deploy an interactive application to evaluate saved models on uploaded labeled test data **without retraining**.

## b. Dataset description
**Source:** UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`. There are **569 records**, **30 numerical features**, and a binary target. We recode the dataset so **1 = malignant** and **0 = benign**. The same stratified 80/20 train/test split (`random_state=42`) is used for all five classifiers. The data is intended for educational ML benchmarking, **not medical decision-making**.

## c. GitHub Repository Link
**TO BE FILLED AFTER YOU CREATE AND PUSH YOUR OWN GITHUB REPOSITORY:** `https://github.com/<your-account>/<your-repository>`

## d. Models used and evaluation
1. Logistic Regression (with StandardScaler)
2. Decision Tree Classifier (max_depth=5, min_samples_leaf=3)
3. K-Nearest Neighbors (k=7, with StandardScaler)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel, with StandardScaler)

Metrics: accuracy, ROC AUC (using predicted probabilities), precision, recall, F1, and Matthews Correlation Coefficient (MCC). All binary classification metrics treat **malignant (1)** as positive. Saved models include their preprocessing steps where applicable.

### Model comparison table
These are reproducible reference-run results from `python train_models.py`; re-run on BITS Virtual Lab and confirm the results in your own environment.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations about model performance
Fill these after inspecting your measured results and confusion matrices. Explain the performance of **each of the five models**, any tradeoffs (especially recall for the malignant class), and identify the overall winner with a defensible metric-based justification.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TO BE WRITTEN FROM YOUR RESULTS |
| Decision Tree | TO BE WRITTEN FROM YOUR RESULTS |
| KNN | TO BE WRITTEN FROM YOUR RESULTS |
| Naive Bayes | TO BE WRITTEN FROM YOUR RESULTS |
| Support Vector Machine | TO BE WRITTEN FROM YOUR RESULTS |
| Overall winner | TO BE SELECTED FROM YOUR RESULTS |

## Run and reproduce

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload the generated `test_data.csv` in the app. Switch between the five saved models to view six metrics, a confusion matrix, and classification report. For a fair comparison, the included test CSV is a **held-out split** never used for fitting. Do not train on the uploaded CSV.

## Streamlit deployment
1. Commit the project including `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `metadata.json`, `model_comparison.csv`, and all five `.joblib` files under `model/`.
2. Go to https://streamlit.io/cloud, sign in with GitHub, choose **New app**, select your repo and branch, choose `app.py`, and deploy.
3. Test the public URL with `test_data.csv` and all five model options.

**Live app link (fill after deployment):** `https://<your-app>.streamlit.app`

## BITS Virtual Lab evidence and final PDF
The assignment requires actual execution in BITS Virtual Lab and **one authentic screenshot**. Submit **one PDF** containing, in order: (1) GitHub repo link, (2) live Streamlit app link, (3) BITS Virtual Lab screenshot, (4) README content, including model metrics, observations, and overall winner. Deadline: **8 November 2026, 11:59 PM**.

## Academic integrity
The assignment states AI tools may be used for learning support, not direct copy-paste submissions. Make your own meaningful design and experiment choices, and write your own interpretation of the results. GitHub history and identical outputs can be checked.
