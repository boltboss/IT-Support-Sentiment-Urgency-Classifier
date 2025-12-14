# 🚀 IT Support Ticket Sentiment & Urgency Classification

An end-to-end NLP-based machine learning system to classify sentiment and urgency
from IT support tickets and generate a final priority score for automated ticket
triaging.

## Features
- Weakly supervised sentiment labeling
- TF-IDF + Logistic Regression models
- Separate sentiment and urgency classifiers
- Combined priority scoring logic
- Interpretable and production-style ML pipeline

## Tech Stack
- Python
- Pandas, NumPy
- NLTK (VADER)
- Scikit-learn
- TF-IDF
- Joblib

## Project Workflow
1. Data preprocessing & language filtering
2. Weak sentiment labeling
3. Exploratory Data Analysis (EDA)
4. Sentiment model training
5. Urgency model training
6. Combined priority scoring

## Results
- Sentiment classification accuracy: ~81%
- Urgency classification accuracy: ~59% (text-only baseline)

## Future Improvements
- Include SLA and system metadata
- Multilingual modeling
- Transformer-based NLP models
- Real-time API deployment
