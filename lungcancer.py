# -*- coding: utf-8 -*-
"""
Lung-Cancer Survival Predictor · Categorized Inputs · krish solanki · 2025
"""

import streamlit as st
import numpy as np
import pickle


MODEL_PATH = "lungcancer_model.sav"
loaded_model = pickle.load(open(MODEL_PATH, "rb"))


def lungcancer_prediction(features: list) -> str:
    pred = loaded_model.predict(np.asarray(features).reshape(1, -1))[0]
    return "Survived ✅" if pred == 1 else "Not Survived ❌"


def main():
    st.set_page_config(page_title="🫁 Lung-Cancer Predictor", page_icon="🫁")
    st.title("🫁 Lung-Cancer Survival Predictor")
    st.markdown("Estimate a patient’s **5-year survival outcome** based on clinical and lifestyle factors.")
    st.write("---")

   
    col1, col2, col3 = st.columns(3, gap="large")

    # Section 1 — Patient Metrics
    with col1:
        st.subheader("🧮 Patient Metrics")
        age = st.number_input("🎂 Age (years)", 1, 120, 60)
        bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 25.0, step=0.1)
        chol = st.number_input("🩸 Cholesterol (mg/dL)", 80, 400, 180)

    # Section 2 — Clinical & Regional Info
    with col2:
        st.subheader("📊 Clinical & Regional Info")
        gender = st.selectbox("⚧ Gender", ["Male", "Female"])
        country = st.selectbox("🌍 Country", ["USA", "Europe", "Asia", "Other"])
        cancer_stage = st.selectbox("🎗 Cancer Stage", ["I", "II", "III", "IV"])
        treatment_type = st.selectbox("💊 Treatment Type", [
            "Surgery", "Chemotherapy", "Radiation", "Immunotherapy", "Combination"
        ])

    # Section 3 — Risk & Medical History
    with col3:
        st.subheader("🧬 Risk & Medical History")
        family_history = st.radio("👪 Family History of Cancer?", ["No", "Yes"])
        smoking_status = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])
        hypertension = st.radio("💓 Hypertension?", ["No", "Yes"])
        asthma = st.radio("🌬 Asthma?", ["No", "Yes"])
        cirrhosis = st.radio("🍺 Cirrhosis?", ["No", "Yes"])
        other_cancer = st.radio("➕ Other Cancers?", ["No", "Yes"])


    gender_enc = {"Male": 0, "Female": 1}[gender]
    country_enc = {"USA": 0, "Europe": 1, "Asia": 2, "Other": 3}[country]
    stage_enc = {"I": 1, "II": 2, "III": 3, "IV": 4}[cancer_stage]
    fam_hist_enc = {"No": 0, "Yes": 1}[family_history]
    smoke_enc = {"Never": 0, "Former": 1, "Current": 2}[smoking_status]
    hyper_enc = {"No": 0, "Yes": 1}[hypertension]
    asthma_enc = {"No": 0, "Yes": 1}[asthma]
    cirr_enc = {"No": 0, "Yes": 1}[cirrhosis]
    other_ca_enc = {"No": 0, "Yes": 1}[other_cancer]
    treat_enc = {
        "Surgery": 0,
        "Chemotherapy": 1,
        "Radiation": 2,
        "Immunotherapy": 3,
        "Combination": 4
    }[treatment_type]


    if st.button("🔍 Predict Survival"):
        input_vector = [
            age, gender_enc, country_enc, stage_enc, fam_hist_enc,
            smoke_enc, bmi, chol, hyper_enc, asthma_enc,
            cirr_enc, other_ca_enc, treat_enc
        ]
        result = lungcancer_prediction(input_vector)
        st.success(f"**Predicted Outcome:** {result}")

 
if __name__ == "__main__":
    main()
