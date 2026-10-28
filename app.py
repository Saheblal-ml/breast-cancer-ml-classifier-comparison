"""Streamlit evaluation app. Uploaded test data is never used to retrain models."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Classification Model Comparison', page_icon='🔬', layout='wide')
st.title('🔬 Classification Model Comparison')
st.caption('Educational project | Wisconsin Diagnostic Breast Cancer dataset | 1 = malignant, 0 = benign')
st.info('Upload a labeled test CSV. Models are loaded from saved files; the app does not retrain them.')
metadata = json.loads((ROOT / 'metadata.json').read_text())
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
model_name = st.selectbox('Select classifier', model_names)
upload = st.file_uploader('Upload labeled test CSV', type=['csv'])
if upload is None:
    st.write('For testing, upload the included **test_data.csv** file.')
    st.stop()
try:
    data = pd.read_csv(upload)
except Exception as exc:
    st.error(f'Could not read CSV: {exc}')
    st.stop()
required = metadata['feature_names'] + ['target']
missing = [col for col in required if col not in data.columns]
if missing:
    st.error('Missing required columns: ' + ', '.join(missing))
    st.stop()
if data.empty or data[required].isna().any().any():
    st.error('CSV must have at least one row and no missing values in required columns.')
    st.stop()
try:
    X = data[metadata['feature_names']].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(data['target'], errors='raise').astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError('Target must be 0 or 1')
    model = joblib.load(ROOT / 'model' / (model_name.lower().replace(' ', '_') + '.joblib'))
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
except Exception as exc:
    st.error(f'Invalid data or model loading error: {exc}')
    st.stop()

metrics = {
    'Accuracy': accuracy_score(y, pred),
    'AUC': roc_auc_score(y, prob) if y.nunique() == 2 else None,
    'Precision': precision_score(y, pred, zero_division=0),
    'Recall': recall_score(y, pred, zero_division=0),
    'F1': f1_score(y, pred, zero_division=0),
    'MCC': matthews_corrcoef(y, pred),
}
st.subheader(f'Evaluation: {model_name}')
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, f'{value:.4f}' if value is not None else 'N/A')
if metrics['AUC'] is None:
    st.warning('AUC is undefined when the uploaded test data contains only one class.')
left, right = st.columns(2)
with left:
    st.subheader('Confusion matrix')
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0,1]), display_labels=['Benign (0)', 'Malignant (1)']).plot(ax=ax, cmap='Blues', colorbar=False)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader('Classification report')
    st.dataframe(pd.DataFrame(classification_report(y, pred, labels=[0,1], output_dict=True, zero_division=0)).transpose())
with st.expander('Preview uploaded test data'):
    st.dataframe(data.head(20))
