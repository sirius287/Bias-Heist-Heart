import streamlit as st
import pandas as pd
import joblib

from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cardiac Event Risk Predictor",
    page_icon="🫀",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("cardiac_bias_model.json")
    return model


@st.cache_resource
def load_features():
    return joblib.load("cardiac_bias_features.pkl")


model = load_model()
feature_info = load_features()

FEATURES = feature_info["features"]


# ============================================================
# HEADER
# ============================================================

st.title("🫀 Cardiac Event Risk Predictor")

st.markdown(
    """
    ### Model Audit Challenge

    This application uses a machine-learning model to estimate
    cardiac event risk.

    **Your task:** investigate whether the model behaves
    consistently across different patient profiles.

    Try changing individual characteristics while keeping the
    other values constant. Compare predictions across different
    demographic and symptom combinations.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Patient Information")


age = st.sidebar.slider(
    "Age",
    min_value=20,
    max_value=90,
    value=50
)


sex = st.sidebar.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)


cp = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    }[x]
)


st.sidebar.subheader("Symptoms")


shortness_of_breath = st.sidebar.selectbox(
    "Shortness of Breath",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


fatigue = st.sidebar.selectbox(
    "Fatigue",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


nausea = st.sidebar.selectbox(
    "Nausea",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


radiating_pain = st.sidebar.selectbox(
    "Radiating Pain",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)


st.sidebar.subheader("Clinical Measurements")


thalach = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=140
)


trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=130
)


chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=220
)


st.sidebar.subheader("Additional Model Features")


fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)


restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2]
)


exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)


oldpeak = st.sidebar.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=7.0,
    value=1.0,
    step=0.1
)


slope = st.sidebar.selectbox(
    "Slope",
    [1, 2, 3]
)


ca = st.sidebar.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)


thal = st.sidebar.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,

    "shortness_of_breath": shortness_of_breath,
    "fatigue": fatigue,
    "nausea": nausea,
    "radiating_pain": radiating_pain,

    "thalach": thalach,
    "trestbps": trestbps,
    "chol": chol,

    "fbs": fbs,
    "restecg": restecg,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])


# Make absolutely sure the feature order matches training
input_data = input_data[FEATURES]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]


# ============================================================
# MAIN RESULT
# ============================================================

st.subheader("Model Prediction")


col1, col2, col3 = st.columns(3)


with col1:

    if prediction == 1:
        st.error("🔴 HIGH RISK")
    else:
        st.success("🟢 LOW RISK")


with col2:

    st.metric(
        "High-Risk Probability",
        f"{probability:.1%}"
    )


with col3:

    st.metric(
        "Low-Risk Probability",
        f"{1 - probability:.1%}"
    )


st.divider()


# ============================================================
# PATIENT PROFILE
# ============================================================

st.subheader("Patient Profile")


profile_col1, profile_col2 = st.columns(2)


with profile_col1:

    st.write("**Demographics**")

    st.write(
        f"""
        - Age: **{age}**
        - Sex: **{"Female" if sex == 0 else "Male"}**
        - Chest Pain Type: **{cp}**
        """
    )


with profile_col2:

    st.write("**Symptoms**")

    st.write(
        f"""
        - Shortness of Breath: **{"Yes" if shortness_of_breath else "No"}**
        - Fatigue: **{"Yes" if fatigue else "No"}**
        - Nausea: **{"Yes" if nausea else "No"}**
        - Radiating Pain: **{"Yes" if radiating_pain else "No"}**
        """
    )


# ============================================================
# MODEL AUDIT INFORMATION
# ============================================================

st.divider()

st.subheader("🔎 Model Audit")

st.info(
    """
    **Investigation suggestion**

    Try creating two patient profiles that are identical in
    every characteristic except one demographic or symptom
    variable.

    Compare the model's predicted risk.

    Look for unexpected differences in predictions between
    otherwise similar patients.
    """
)


# ============================================================
# RAW MODEL INPUT
# ============================================================

with st.expander("View Model Input"):

    display_data = input_data.copy()

    display_data["sex"] = (
        display_data["sex"]
        .map({0: "Female", 1: "Male"})
    )

    st.dataframe(
        display_data,
        use_container_width=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application is an educational AI model-auditing "
    "challenge. It is not a medical diagnostic tool and should "
    "not be used for clinical decision-making."
)