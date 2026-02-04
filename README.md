✨ STUDENT PERFORMANCE PREDICTOR

A Machine Learning Web App for Predicting Final Exam Scores (G3)

🎯 Project Overview

This project predicts a student's final exam score (G3) using behavioral, demographic, family, and academic data from the UCI Student Performance (Math) dataset.
It aims to help educators identify students who may be at academic risk before the final exam.

🚀 Live Demo

🔗 Streamlit App: https://studentperformancepredictor-ix2kp5fyavjdeyin7umtnt.streamlit.app/

📦 Features

Predict final exam grade G3 (0–20) with ML

Interactive input form with 40+ parameters

Smart recommendations to improve student performance

Model insights:

R², MSE, MAE

Top 10 feature importances

Clean and responsive UI

Ready for deployment and extension

🧠 Machine Learning Approach
✔ Dataset

UCI Student Performance (Math)

395 rows, 33 features

Focus on predicting G3 (final grade)

✔ Preprocessing

Categorical encoding using pd.get_dummies() with drop_first=True

Scaling not required for tree-based models

80/20 train-test split

✔ Models evaluated

Linear Regression

Decision Tree Regressor

Gradient Boosting Regressor

Random Forest Regressor (Best performing)

📈 Deployed Model Summary
Metric	Value
R²	~0.85 (depending on split)
MSE	Low
MAE	Low

(Exact values shown in app’s “Model Insights” section)

🧬 Top Features Affecting G3

G1, G2

studytime

failures

absences

Alcohol consumption levels

Behavioral features (freetime, goout, romantic, etc.)


🛠 Tech Stack

Python

Streamlit

Scikit-Learn

Pandas

NumPy

Matplotlib

Joblib
