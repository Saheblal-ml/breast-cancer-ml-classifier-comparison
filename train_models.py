"""Educational reference: train and evaluate five classifiers on a shared split."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'model'
MODELS.mkdir(exist_ok=True)
raw = load_breast_cancer(as_frame=True)
X = raw.data
# Dataset target: 0 malignant, 1 benign. Recode 1 = malignant for intuitive fraud-like positive metrics.
y = (raw.target == 0).astype(int)
y.name = 'target'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
configs = {
    'Logistic Regression': Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(max_iter=2000, random_state=42))]),
    'Decision Tree': Pipeline([('model', DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42))]),
    'KNN': Pipeline([('scale', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=7))]),
    'Naive Bayes': Pipeline([('model', GaussianNB())]),
    'Support Vector Machine': Pipeline([('scale', StandardScaler()), ('model', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))]),
}
rows = []
for name, pipeline in configs.items():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    rows.append({
        'ML Model Name': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'AUC': roc_auc_score(y_test, probabilities),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0),
        'MCC': matthews_corrcoef(y_test, predictions),
    })
    joblib.dump(pipeline, MODELS / (name.lower().replace(' ', '_') + '.joblib'))

pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
results = pd.DataFrame(rows)
results.to_csv(ROOT / 'model_comparison.csv', index=False)
(ROOT / 'metadata.json').write_text(json.dumps({'feature_names': list(X.columns), 'target': 'target', 'positive_class': 'malignant (1)', 'negative_class': 'benign (0)', 'dataset': 'UCI Breast Cancer Wisconsin Diagnostic via scikit-learn'}, indent=2))
print(results.round(4).to_string(index=False))
print('Saved trained pipelines, test_data.csv, metadata.json and model_comparison.csv')
