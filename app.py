import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAD Susceptibility Predictor", layout="wide")
st.title("CAD Susceptibility Predictor")


# ===================================
# LOAD FILE
# ===================================
def load_data(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


# ===================================
# FIND TARGET COLUMN
# ===================================
def find_target_col(df):
    for col in df.columns:
        if "cad" in col.lower():
            return col
    st.error("No column containing 'CAD' found.")
    st.stop()


# ===================================
# BUILD PIPELINE
# ===================================
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

    pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    return pipe


# ===================================
# SESSION STATE
# ===================================
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "meta" not in st.session_state:
    st.session_state.meta = {}
if "df" not in st.session_state:
    st.session_state.df = None


# ===================================
# UPLOAD DATA
# ===================================
uploaded = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload your dataset to start.")
    st.stop()

df = load_data(uploaded)
df.columns = [str(c).strip() for c in df.columns]
st.session_state.df = df

st.write(f"Dataset loaded. Total rows: {df.shape[0]}")
st.dataframe(df.head())


# ===================================
# TRAIN MODEL
# ===================================
target_col = find_target_col(df)

if st.button("Train Model"):
    with st.spinner("Training..."):

        data = df.dropna(subset=[target_col]).copy()
        data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
        data = data.dropna(subset=[target_col])
        data[target_col] = data[target_col].astype(int)

        if data[target_col].nunique() < 2:
            st.error("CAD column must have at least 2 classes.")
            st.stop()

        X = data.drop(columns=[target_col])
        y = data[target_col]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write("Numeric columns:", numeric_cols)
        st.write("Categorical columns:", categorical_cols)

        pipeline = build_pipeline(numeric_cols, categorical_cols)

        # Train-test split
        if len(data) < 5:
            st.warning("Dataset too small for split. Training on full dataset.")
            pipeline.fit(X, y)
            X_train = X
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42)
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            try:
                probs = pipeline.predict_proba(X_test)[:, 1]
                st.write("ROC AUC:", roc_auc_score(y_test, probs))
            except:
                pass

            st.text(classification_report(y_test, preds, zero_division=0))

        # Save in session
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]

        try:
            feature_names = pre.get_feature_names_out()
        except:
            feature_names = [f"f{i}" for i in range(len(clf.feature_importances_))]

        st.session_state.model = pipeline
        st.session_state.meta = {
            "pre": pre,
            "clf": clf,
            "feature_names": feature_names,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols
        }
        st.session_state.trained = True

        st.success("Model trained successfully!")

        # ---------------------------
        # Feature Importance Chart
        # ---------------------------
        st.subheader("🔧 Feature Importance (Model-Based)")

        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(7, 4))
        plt.barh(np.array(feature_names)[sorted_idx][::-1],
                 importances[sorted_idx][::-1])
        plt.title("Top Feature Importances")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()


# ===================================
# PREDICTION SECTION
# ===================================
if st.session_state.trained:

    st.header("Make a Prediction")

    meta = st.session_state.meta
    model = st.session_state.model

    # ---------------------------
    # Option 1: Select row
    # ---------------------------
    st.subheader("Option 1: Auto-fill from Dataset Row")

    prefill = {}
    use_row = st.checkbox("Use a row from dataset")

    if use_row:
        idx = st.number_input(
            f"Select index (0 - {df.shape[0]-1})",
            min_value=0, max_value=df.shape[0]-1, value=0
        )
        if st.button("Load Row"):
            row = df.iloc[int(idx)]
            for col in meta["numeric_cols"] + meta["categorical_cols"]:
                if col in df.columns:
                    prefill[col] = row[col]
            st.success(f"Row {idx} loaded!")


    # ---------------------------
    # Option 2: Manual Entry
    # ---------------------------
    st.subheader("Option 2: Manual Entry")

    input_dict = {}
    c1, c2 = st.columns(2)

    for col in meta["numeric_cols"]:
        input_dict[col] = c1.text_input(
            f"{col} (numeric)", value=str(prefill.get(col, "")))

    for col in meta["categorical_cols"]:
        input_dict[col] = c2.text_input(
            f"{col} (text)", value=str(prefill.get(col, "")))

    # ---------------------------
    # Predict Button
    # ---------------------------
    if st.button("Predict"):

        df_input = pd.DataFrame([input_dict])

        # Convert numeric
        for col in meta["numeric_cols"]:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        st.success(f"Prediction: {'CAD Susceptible (1)' if pred==1 else 'Not Susceptible (0)'}")
        st.info(f"Probability: {prob:.3f}")
