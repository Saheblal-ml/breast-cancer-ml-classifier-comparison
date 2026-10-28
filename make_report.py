from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
ROOT=Path(__file__).resolve().parent
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='SmallCustom',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallMonoCustom',parent=styles['BodyText'],fontName='Courier',fontSize=7.5,leading=10))
doc=SimpleDocTemplate(str(ROOT/'submission_draft.pdf'),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48)
story=[]
def p(s,style='SmallCustom'): story.append(Paragraph(s,styles[style]))
def h(s): p(s,'Heading2')
p('Machine Learning Assignment 2 - Submission Draft','Title')
p('<b>IMPORTANT: This is NOT a complete submission.</b> Replace link placeholders, add a genuine BITS Virtual Lab screenshot, review the README, and validate all results on your own lab environment.')
h('1. GitHub Repository Link')
p('[ADD YOUR ACTUAL PUBLIC GITHUB REPOSITORY URL HERE]')
h('2. Live Streamlit App Link')
p('[ADD YOUR ACTUAL WORKING STREAMLIT COMMUNITY CLOUD URL HERE]')
h('3. BITS Virtual Lab Execution Screenshot')
p('[INSERT YOUR OWN SCREENSHOT FROM EXECUTION ON BITS VIRTUAL LAB HERE]')
story.append(PageBreak())
h('4. README Content')
h('a. Problem statement')
p('Compare five supervised classification models on one dataset using six metrics, then deploy a Streamlit app that loads saved trained models and evaluates uploaded labeled test data without retraining.')
h('b. Dataset description')
p('UCI Breast Cancer Wisconsin (Diagnostic), accessed through scikit-learn: 569 rows, 30 numeric input features, binary target recoded as 1 = malignant and 0 = benign. Stratified 80/20 split with random_state=42. Educational benchmarking only, not medical diagnosis.')
h('c. GitHub Repository Link')
p('[ADD YOUR ACTUAL GITHUB URL]')
h('d. Models used and comparison table')
p('Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, Support Vector Machine. AUC is computed from predicted probabilities; precision/recall/F1 use malignant (1) as positive.')
r=pd.read_csv(ROOT/'model_comparison.csv')
columns=list(r.columns)
data=[['Model','Accuracy','AUC','Precision','Recall','F1','MCC']]
for _,row in r.iterrows():
    data.append([str(row['ML Model Name']).replace('Support Vector Machine','SVM').replace('Logistic Regression','Logistic Reg.')]+[f'{row[k]:.4f}' for k in columns[1:]])
t=Table(data,colWidths=[105]+[65]*6,repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#193653')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#CBD5E1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F9')]),('FONTSIZE',(0,0),(-1,-1),7.6),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8)]))
story.append(t); story.append(Spacer(1,16))
p('Scores above were computed in a local reference run. Reproduce and verify them in BITS Virtual Lab before submitting.')
h('Observations about model performance - student review required')
for name in r['ML Model Name']:
    p(f'<b>{name}:</b> [Write your own interpretation based on metrics and confusion matrix.]')
p('<b>Overall winner:</b> [Choose and justify using the metrics relevant to your classification objective.]')
h('How to run')
p('Install dependencies from requirements.txt, execute train_models.py, run streamlit run app.py, upload test_data.csv and switch among the five models.')
h('Academic integrity and final checks')
p('Review and customize the code, document independent decisions and experiments, verify GitHub and Streamlit links, include one genuine lab screenshot, and submit the final PDF by 8 November 2026, 11:59 PM.')
doc.build(story)
print(ROOT/'submission_draft.pdf')
