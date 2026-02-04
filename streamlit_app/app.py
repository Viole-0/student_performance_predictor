import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# 📂 Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "trained_feature_names.pkl")
PERF_PATH = os.path.join(BASE_DIR, "..", "models", "model_performance_summary.csv")
IMP_PATH = os.path.join(BASE_DIR, "..", "models", "feature_importances_rf.csv")

# -----------------------------
# 💾 Load model + feature names
# -----------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
except Exception as e:
    st.error(f"❌ Model or feature metadata file not found.\nDetails: {e}")
    st.stop()

# -----------------------------
# 🧭 Sidebar Navigation
# -----------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "🔮 Predict Grade",
        "📊 Model Insights",
        "👥 About & Credits"
    ]
)

# -----------------------------
# ℹ️ Common Project Info (Sidebar)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
**PREDICTIVE MODELLING OF STUDENT SUCCESS USING MACHINE LEARNING AND BEHAVIORAL DATA**

- Dataset: UCI Student Performance (Math)  
- Target: Final grade **G3** (0–20)  
- Goal: Early identification of students who may be at risk.
""")


# ============================================================
# 🔮 PAGE 1 – PREDICT GRADE
# ============================================================
if page == "🔮 Predict Grade":
    st.title("🎓 Student Performance Predictor")
    st.markdown(
        "Predict a student's **final exam score (G3)** using features from the "
        "UCI Student Performance dataset."
    )

    st.header("📋 Enter Student Details")

    # -----------------------------
    # 🧮 Input Form
    # -----------------------------
    with st.form("student_form"):
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            school = st.selectbox("🏫 School", ["GP", "MS"])
            sex = st.selectbox("🚻 Sex", ["F", "M"])
            age = st.slider("🎂 Age", 15, 22, 17)
            address = st.selectbox("🏙️ Address", ["U", "R"])
            famsize = st.selectbox("👨‍👩‍👧‍👦 Family Size", ["LE3", "GT3"])
            Pstatus = st.selectbox("🏠 Parent Cohabitation Status", ["T", "A"])
            Medu = st.slider("👩‍🎓 Mother's Education (0–4)", 0, 4, 2)
            Fedu = st.slider("👨‍🎓 Father's Education (0–4)", 0, 4, 2)

        with col2:
            Mjob = st.selectbox("👩 Mother's Job", ["teacher", "health", "services", "at_home", "other"])
            Fjob = st.selectbox("👨 Father's Job", ["teacher", "health", "services", "at_home", "other"])
            reason = st.selectbox("📌 Reason for School Choice", ["home", "reputation", "course", "other"])
            guardian = st.selectbox("🧑‍🧒 Guardian", ["mother", "father", "other"])
            traveltime = st.slider("🚌 Travel Time (1–4)", 1, 4, 1)
            studytime = st.slider("📖 Weekly Study Time (1–4)", 1, 4, 2)
            failures = st.slider("❌ Past Failures (0–4)", 0, 4, 0)

        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            schoolsup = st.selectbox("📚 Extra Educational Support (schoolsup)", ["no", "yes"])
            famsup = st.selectbox("👨‍👩‍👧 Family Support (famsup)", ["no", "yes"])
            paid = st.selectbox("💰 Extra Paid Classes (paid)", ["no", "yes"])
            activities = st.selectbox("⚽ Extracurricular Activities", ["no", "yes"])
            nursery = st.selectbox("🏡 Attended Nursery School", ["no", "yes"])
            higher = st.selectbox("🎓 Wants Higher Education", ["no", "yes"])
            internet = st.selectbox("🌐 Internet Access at Home", ["no", "yes"])
            romantic = st.selectbox("❤️ In a Romantic Relationship", ["no", "yes"])

        with col4:
            famrel = st.slider("🏡 Family Relationship Quality (1–5)", 1, 5, 4)
            freetime = st.slider("🕒 Free Time (1–5)", 1, 5, 3)
            goout = st.slider("🎉 Going Out (1–5)", 1, 5, 3)
            Dalc = st.slider("🍺 Workday Alcohol (1–5)", 1, 5, 1)
            Walc = st.slider("🍻 Weekend Alcohol (1–5)", 1, 5, 2)
            health = st.slider("💊 Health (1–5)", 1, 5, 3)
            absences = st.slider("🚫 Absences", 0, 93, 5)
            G1 = st.slider("📘 First Period Grade (G1)", 0, 20, 10)
            G2 = st.slider("📗 Second Period Grade (G2)", 0, 20, 10)

        submitted = st.form_submit_button("🔮 Predict Final Grade (G3)")

    if submitted:
        # -----------------------------
        # 🧮 Build raw input DataFrame
        # -----------------------------
        raw_input = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
            "G1": G1,
            "G2": G2,
        }

        input_df = pd.DataFrame([raw_input])

        # -----------------------------
        # 🔣 Encoding – mirror notebook (get_dummies + drop_first)
        # -----------------------------
        df_encoded = pd.get_dummies(input_df, drop_first=True)
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

        # -----------------------------
        # 📈 Predict + Interpretation
        # -----------------------------
        try:
            prediction = model.predict(df_encoded)[0]
            prediction = max(0, min(20, prediction))   # clamp to [0, 20]

            st.success(f"📈 Predicted Final Exam Score (G3): **{prediction:.2f} / 20**")

            # Interpretation bands
            if prediction >= 15:
                st.info("🌟 This student is likely to perform **very well** in the final exam.")
            elif prediction >= 10:
                st.info("👍 This student is likely to **pass comfortably**, with room for improvement.")
            elif prediction >= 8:
                st.warning("⚠️ This student may be at **borderline risk**. Extra support could help secure a pass.")
            else:
                st.error("🚨 This student appears to be at **high risk of underperforming**. Early intervention is recommended.")

            # -----------------------------
            # 🧭 Personalized Suggestions
            # -----------------------------
            suggestions = []

            if studytime <= 2:
                suggestions.append("📖 Increasing study time to at least **3–4 units/week** may help improve performance.")
            if failures > 0:
                suggestions.append("🔁 Focus on subjects previously failed, possibly with extra tutoring or revision classes.")
            if absences > 10:
                suggestions.append("🏫 Reducing absences can significantly affect performance. Aim for more consistent attendance.")
            if goout >= 4:
                suggestions.append("🎉 Balancing social activities with academics could improve focus and outcomes.")
            if Walc >= 3 or Dalc >= 3:
                suggestions.append("🍺 Reducing alcohol intake may help improve concentration and consistency in studies.")
            if internet == "no":
                suggestions.append("🌐 Consider arranging better access to study resources (online or offline alternatives).")
            if prediction < 10:
                suggestions.append("🆘 A structured study plan with regular revision and teacher support is strongly recommended.")

            if suggestions:
                st.markdown("---")
                st.subheader("🧠 Smart Recommendations for Improvement")
                for item in suggestions:
                    st.markdown(f"- {item}")
            else:
                st.success("✨ No major risk indicators detected. Keep up the great work!")

        except Exception as e:
            st.error(f"❌ Something went wrong during prediction: {e}")


# ============================================================
# 📊 PAGE 2 – MODEL INSIGHTS
# ============================================================
elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Explainability")

    # -----------------------------
    # Performance Summary
    # -----------------------------
    st.subheader("📌 Model Performance on Test Set")
    try:
        perf_df = pd.read_csv(PERF_PATH)

        # Handle 'R²' vs 'R2'
        if "R2" not in perf_df.columns and "R²" in perf_df.columns:
            perf_df = perf_df.rename(columns={"R²": "R2"})

        st.dataframe(perf_df)

        # Highlight Random Forest row
        rf_row = perf_df.loc[perf_df["Model"] == "Random Forest"].iloc[0]
        st.markdown(f"""
**Deployed Model:** Random Forest Regressor  

- R²: `{rf_row["R2"]:.2f}`  
- MSE: `{rf_row["MSE"]:.2f}`  
- MAE: `{rf_row["MAE"]:.2f}`  
""")

    except Exception as e:
        st.info("ℹ️ Model performance summary not available.")
        # st.write(e)

    st.markdown("---")

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("🧬 Top Influential Features (Random Forest)")

    try:
        imp_df = pd.read_csv(IMP_PATH).sort_values(by="importance", ascending=False)
        top_k = imp_df.head(10)

        st.markdown("**Top 10 Features by Importance:**")
        st.dataframe(top_k.reset_index(drop=True))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_k["feature"][::-1], top_k["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Most Important Features (Random Forest)")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
These features contribute the most to the model's prediction of **G3**.  
Typically, previous grades (`G1`, `G2`), study time, failures, and absences
have strong influence on student performance.
""")

    except Exception as e:
        st.info("ℹ️ Feature importance data not available.")
        # st.write(e)


# ============================================================
# 👥 PAGE 3 – ABOUT & CREDITS
# ============================================================
elif page == "👥 About & Credits":
    st.title("👥 About the Project & Team")

    st.subheader("📘 Project Overview")
    st.markdown("""
This project, **PREDICTIVE MODELLING OF STUDENT SUCCESS USING MACHINE LEARNING AND BEHAVIORAL DATA**, uses machine learning
to predict a student's final exam grade (**G3**) based on demographic, family,
academic, and behavioral factors from the **UCI Student Performance (Math)** dataset.

The goal is to help identify students who may be at risk **before** the final exam,
so that teachers and institutions can plan early interventions.
""")

    st.subheader("🧠 Methodology (High Level)")
    st.markdown("""
1. **Data Collection** – UCI Student Performance dataset (Math).  
2. **Preprocessing** – Handling missing values, encoding categorical features using
   `pd.get_dummies(..., drop_first=True)`.  
3. **Modeling** – Trained and compared multiple regression models:
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  
4. **Model Selection** – Random Forest achieved the best R² score and was selected
   as the final deployed model.  
5. **Deployment** – Model and feature metadata were saved using `joblib`, and a
   Streamlit web app was built for interactive prediction.
""")
    
    st.subheader("🚧 Limitations & Future Work")
    st.markdown("""
- The model is trained on a single dataset from a specific context (not globally representative).  
- It predicts numeric grade (**regression**) but does not explicitly classify risk levels
  (e.g., low/medium/high risk) in the backend model.  
- Future work could include:
  - Adding classification models for risk categorization.  
  - Using explainable AI techniques (e.g., SHAP) for per-student explanations.  
  - Extending the dataset to multiple schools and academic years.
""")
