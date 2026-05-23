import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# ---------------- 1. Page configuration ----------------
st.set_page_config(
    page_title="AKI Risk Prediction System",
    layout="wide",
    page_icon="🏥"
)

# ---------------- 2. Load model ----------------
@st.cache_resource
def load_model_and_meta():
    model = CatBoostClassifier()
    model.load_model("CatBoost_model_py.cbm")  # The model file should be in the same directory
    expected_features = model.feature_names_
    cat_features = model.get_cat_feature_indices()
    return model, expected_features, cat_features

try:
    model, expected_features, cat_indices = load_model_and_meta()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# ---------------- 3. Sidebar input controls ----------------
st.sidebar.header("Input Clinical Parameters")

def get_input():
    inputs = {}

    # Continuous variables
    inputs["BUN"] = st.sidebar.number_input("BUN (mg/dL)", value=20.0, step=1.0)
    inputs["Cr"] = st.sidebar.number_input("Creatinine, Cr (mg/dL)", value=1.0, step=0.1)
    inputs["SOFA"] = st.sidebar.number_input("SOFA Score", value=2.0, step=1.0)
    inputs["Lactate"] = st.sidebar.number_input("Lactate (mmol/L)", value=2.0, step=0.1)
    inputs["RR"] = st.sidebar.number_input("Respiratory Rate, RR (breaths/min)", value=20, step=1)
    inputs["Alb"] = st.sidebar.number_input("Albumin, Alb (g/dL)", value=3.5, step=0.1)
    inputs["Glu"] = st.sidebar.number_input("Glucose, Glu (mmol/L)", value=5.5, step=0.1)
    inputs["Weight"] = st.sidebar.number_input("Body Weight (kg)", value=65.0, step=1.0)
    inputs["SpO2"] = st.sidebar.number_input("Oxygen Saturation, SpO₂ (%)", value=98, step=1)

    # Binary variables
    inputs["T2DM"] = str(int(st.sidebar.checkbox("Type 2 Diabetes Mellitus (T2DM)", value=False)))
    inputs["HF"] = str(int(st.sidebar.checkbox("Heart Failure (HF)", value=False)))

    # Generate DataFrame strictly according to the feature order used during model training
    df = pd.DataFrame([inputs])[expected_features]
    return df

input_df = get_input()

# ---------------- 4. Main interface ----------------
st.title("🏥 AKI Risk Prediction for Elderly Patients with Intestinal Obstruction")
st.write(
    "Enter the patient's clinical parameters in the sidebar, "
    "then click the button below to obtain the AI-based AKI risk assessment "
    "and model explanation."
)

if st.button("Start Prediction", type="primary"):
    try:
        # Predict probability
        prob = model.predict_proba(input_df)[0][1]

        # Result dashboard
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Prediction Result")
            st.metric("Predicted AKI Risk", f"{prob:.2%}")

            # Risk stratification
            if prob > 0.7:
                st.error("⚠️ High Risk")
            elif prob > 0.3:
                st.warning("⚠️ Medium Risk")
            else:
                st.success("✅ Low Risk")

            st.write("**Current Input Parameters:**")
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

        with col2:
            st.subheader("Model Explanation Using SHAP")

            # SHAP explanation
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            # Plot SHAP waterfall plot
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)

            st.info(
                "💡 Red bars indicate features that increase the predicted AKI risk, "
                "whereas blue bars indicate features that decrease the predicted AKI risk."
            )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info(
            "Please check whether the input data types are consistent with those used during model training."
        )

# ---------------- 5. Developer debug information ----------------
with st.expander("🔍 Developer Debug Information"):
    st.write("Expected feature order of the model:", expected_features)
    st.write("Categorical feature indices:", cat_indices)
    st.write("Current input DataFrame:", input_df)

# ---------------- 6. Footer ----------------
st.markdown("---")
st.caption(
    "Note: This tool is intended for clinical research purposes only and should not be used as the sole basis for clinical diagnosis or treatment decisions."
)