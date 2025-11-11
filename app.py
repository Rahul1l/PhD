# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def load_excel_or_csv(uploaded_file):
    """Loads an uploaded CSV or Excel file into a pandas DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

def find_target_column(df):
    """Automatically detects the target/label column."""
    lower_cols = [c.lower() for c in df.columns.astype(str)]
    for cand in ["cvd", "target", "disease", "label", "outcome"]:
        if cand in lower_cols:
            return df.columns[lower_cols.index(cand)]
    # if not found, check for 0/1 binary column
    for c in df.columns:
        vals = df[c].dropna()
        if vals.nunique() == 2:
            return c
    return df.columns[-1]

def build_pipeline(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    return model

def train_model(df):
    """Train model from uploaded DataFrame and return trained object."""
    target_col = find_target_column(df)
    st.write(f"✅ Target column detected: `{target_col}`")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    model = build_pipeline(numeric_cols, categorical_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    try:
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None

    st.write("### Model Performance on Test Split")
    st.text(classification_report(y_test, preds))
    if auc:
        st.write(f"ROC-AUC Score: **{auc:.3f}**")

    return model, numeric_cols, categorical_cols, X_train, y_train

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Disease Susceptibility Checker", layout="centered")
st.title("🩺 Disease Susceptibility Prediction App")

st.markdown("""
Upload your medical dataset (CSV/XLSX) below.  
The app will train a model and let you check if a patient is susceptible to the disease.
""")

uploaded_file = st.file_uploader("📤 Upload dataset (CSV or Excel)", type=["csv", "xlsx"])
model = None
numeric_cols = []
categorical_cols = []
X_train = None
y_train = None

if uploaded_file:
    df = load_excel_or_csv(uploaded_file)
    st.success(f"✅ Loaded dataset with shape: {df.shape}")
    st.dataframe(df.head())

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            model, numeric_cols, categorical_cols, X_train, y_train = train_model(df)
            st.success("Model trained successfully!")

if model:
    st.markdown("---")
    st.subheader("🔍 Enter Patient Details")

    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(numeric_cols):
        box = cols[i % 2].text_input(col, "")
        input_data[col] = box

    for col in categorical_cols:
        input_data[col] = st.text_input(col, "")

    if st.button("🧾 Predict"):
        df_input = pd.DataFrame([input_data])
        for c in numeric_cols:
            try:
                df_input[c] = pd.to_numeric(df_input[c])
            except Exception:
                df_input[c] = np.nan

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

        result = "🩸 Susceptible" if int(pred) == 1 else "✅ Not Susceptible"
        st.write("### Prediction Result:", result)
        if prob is not None:
            st.write(f"**Probability of Disease:** {prob:.2f}")

        # SHAP explanation
        with st.expander("📊 Explain this Prediction (SHAP)"):
            try:
                pre = model.named_steps["pre"]
                clf = model.named_steps["clf"]
                X_train_trans = pre.transform(X_train)
                explainer = shap.TreeExplainer(clf, data=X_train_trans)
                shap_values = explainer.shap_values(pre.transform(df_input))

                plt.figure(figsize=(8,4))
                shap.waterfall_plot(shap.Explanation(values=shap_values[1][0],
                                                    base_values=explainer.expected_value[1],
                                                    feature_names=pre.get_feature_names_out(),
                                                    data=pre.transform(df_input)[0]))
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {e}")
