import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# ---------- Aesthetics ----------
st.markdown(
    """
    <style>
    .main {
        padding: 2rem 2rem;
    }
    .stApp {
        background: linear-gradient(180deg, rgba(250,250,250,1) 0%, rgba(240,245,255,1) 100%);
    }
    .title {
        font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; text-align: center;
    }
    .subtitle {
        color: #555; margin-bottom: 1.5rem; text-align: center;
    }
    .card {
        background: white; border-radius: 1rem; padding: 1.2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    .footer-note {
        color: #666; font-size: 0.9rem; margin-top: 1rem;
    }
    .positive {
        font-size: 1.5rem; font-weight: bold; color: red;
    }
    .negative {
        font-size: 1.5rem; font-weight: bold; color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🩺 Diabetes Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details to check whether the person has diabetes or not.</div>', unsafe_allow_html=True)

# ---------- Load model ----------
MODEL_PATH = "diabetes_best_model.pkl"
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipe = load_model()

# Expected feature columns
feature_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                   "BMI", "DiabetesPedigreeFunction", "Age"]

# ---------- Input fields (center) ----------
st.markdown("### ✍️ Enter Patient Details")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)

    with col2:
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.0, step=0.1)

    with col3:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.47, step=0.01)
        Age = st.number_input("Age", min_value=10, max_value=100, value=33, step=1)

    submitted = st.form_submit_button("Check Diabetes Status")

input_data = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age
}

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if submitted:
        X_in = pd.DataFrame([input_data])
        proba = pipe.predict_proba(X_in)[0, 1]
        pred = int(pipe.predict(X_in)[0])

        if pred == 1:
            st.markdown(f"<div class='positive'>🚨 The person is likely to have Diabetes! (Probability: {proba:.2%})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='negative'>✅ The person is NOT likely to have Diabetes. (Probability: {proba:.2%})</div>", unsafe_allow_html=True)

        st.progress(float(proba))

        # Feature influence explanation
        st.subheader("🔍 Feature Influence")
        model = pipe.named_steps["model"]
        feature_names = feature_columns

        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            influence = pd.DataFrame({"Feature": feature_names, "Weight": coefs})
            influence = influence.reindex(influence.Weight.abs().sort_values(ascending=False).index)
            st.bar_chart(influence.set_index("Feature"))
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            influence = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            influence = influence.sort_values("Importance", ascending=False)
            st.bar_chart(influence.set_index("Feature"))
        else:
            st.info("Feature importance not available for this model.")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About the Model")
    st.write("This model is a scikit-learn pipeline that includes:")
    st.markdown("- Zero-value imputation (median) for appropriate columns")
    st.markdown("- Standardization with StandardScaler")
    st.markdown("- Best estimator selected during CV")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Optional: Quick EDA ----------
eda_path = "diabetes.csv"
if os.path.exists(eda_path):
    st.markdown("### 📊 Quick EDA (using local diabetes.csv)")
    ddf = pd.read_csv(eda_path)
    st.write("Shape:", ddf.shape)
    st.dataframe(ddf.head())

    if "Outcome" in ddf.columns:
        counts = ddf["Outcome"].value_counts().sort_index()
        fig = plt.figure()
        plt.bar([str(i) for i in counts.index], counts.values)
        plt.title("Class Balance")
        plt.xlabel("Outcome")
        plt.ylabel("Count")
        st.pyplot(fig)

st.markdown('<div class="footer-note">Note: This app is for educational purposes only and not a medical device.</div>', unsafe_allow_html=True)
