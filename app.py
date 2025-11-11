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
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAD Susceptibility Predictor", layout="wide")
st.title("CAD Susceptibility Predictor (Upload Your Dataset)")


# -------------------------------
# LOAD FILE
# -------------------------------
def load_data(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


# -------------------------------
# FIND TARGET COLUMN (CAD)
# -------------------------------
def find_target_col(df):
    for col in df.columns:
        if "cad" in col.lower():
            return col
    st.error("No CAD column found. Ensure the CAD column contains 'CAD' in its name.")
    st.stop()


# -------------------------------
# BUILD PIPELINE
# -------------------------------
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
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    return pipe


# -------------------------------
# SESSION VARIABLES
# -------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "meta" not in st.session_state:
    st.session_state.meta = {}
if "df" not in st.session_state:
    st.session_state.df = None


# -------------------------------
# UPLOAD DATA
# -------------------------------
uploaded = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload your dataset to continue.")
    st.stop()

df = load_data(uploaded)
df.columns = [str(c).strip() for c in df.columns]
st.session_state.df = df

st.write("Dataset loaded. Total rows:", df.shape[0])
st.dataframe(df.head())


# -------------------------------
# TRAIN MODEL
# -------------------------------
target_col = find_target_col(df)

if st.button("Train Model"):
    with st.spinner("Training model..."):

        # Clean target
        data = df.dropna(subset=[target_col]).copy()
        data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
        data = data.dropna(subset=[target_col])
        data[target_col] = data[target_col].astype(int)

        if data[target_col].nunique() < 2:
            st.error("Target column must contain at least 2 classes.")
            st.stop()

        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Detect numeric + categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write("Numeric columns:", numeric_cols)
        st.write("Categorical columns:", categorical_cols)

        pipeline = build_pipeline(numeric_cols, categorical_cols)

        # Safe train-test split
        if data.shape[0] < 5:
            st.warning("Dataset too small for split. Training on full data.")
            pipeline.fit(X, y)
            X_train = X.copy()
            y_train = y.copy()
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

        # Save metadata
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]

        X_train_trans = pre.transform(X_train)
        try:
            X_train_trans = X_train_trans.toarray()
        except:
            X_train_trans = np.array(X_train_trans)

        try:
            feature_names = pre.get_feature_names_out()
        except:
            feature_names = [f"f{i}" for i in range(X_train_trans.shape[1])]

        st.session_state.model = pipeline
        st.session_state.meta = {
            "pre": pre,
            "clf": clf,
            "X_train_raw": X_train,
            "X_train_trans": X_train_trans,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": feature_names
        }
        st.session_state.trained = True

        st.success("Model trained successfully!")


# -----------------------------------
# PREDICTION UI
# -----------------------------------
if st.session_state.trained:

    st.header("Make a Prediction")

    model = st.session_state.model
    meta = st.session_state.meta

    # -------------------------------
    # OPTION 1: SELECT ROW
    # -------------------------------
    st.subheader("⬇ Option 1: Auto-fill inputs from dataset row")
    prefill = {}

    use_row = st.checkbox("Fill inputs using a row from uploaded dataset")

    if use_row:
        row_index = st.number_input(
            f"Row index (0 - {df.shape[0]-1})",
            min_value=0, max_value=df.shape[0]-1, value=0)

        if st.button("Load Row"):
            row = df.iloc[int(row_index)]
            for col in meta["numeric_cols"] + meta["categorical_cols"]:
                if col in df.columns:
                    prefill[col] = row[col]
            st.success(f"Row {row_index} loaded!")


    # -------------------------------
    # OPTION 2: MANUAL ENTRY
    # -------------------------------
    st.subheader("Option 2: Manual Entry (editable even after row selection)")

    input_dict = {}
    col1, col2 = st.columns(2)

    for col in meta["numeric_cols"]:
        default = "" if col not in prefill else str(prefill[col])
        input_dict[col] = col1.text_input(f"{col} (numeric)", value=default)

    for col in meta["categorical_cols"]:
        default = "" if col not in prefill else str(prefill[col])
        input_dict[col] = col2.text_input(f"{col} (text)", value=default)


    # -------------------------------
    # PREDICT
    # -------------------------------
    if st.button("Predict"):
        df_input = pd.DataFrame([input_dict])

        # Convert numeric
        for col in meta["numeric_cols"]:
            try:
                df_input[col] = pd.to_numeric(df_input[col])
            except:
                df_input[col] = np.nan

        pre = meta["pre"]
        clf = meta["clf"]

        # Transform
        inst = pre.transform(df_input)
        inst = inst.toarray() if hasattr(inst, "toarray") else np.array(inst)

        # Prediction
        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        st.success(f"Prediction: {'CAD Susceptible (1)' if pred==1 else 'Not Susceptible (0)'}")
        st.info(f"Probability: {prob:.3f}")

        # -----------------------------------
        # SHAP EXPLANATION (compatible with all SHAP versions)
        # -----------------------------------
        st.subheader("SHAP Explanation")

        try:
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(inst)
            feature_names = meta["feature_names"]

            # Global importance
            st.write("Feature importance (mean |SHAP|):")
            plt.figure(figsize=(7, 4))

            if isinstance(shap_vals, list):
                sv = shap_vals[1]  # class 1 for binary classification
            else:
                sv = shap_vals

            mean_abs = np.mean(np.abs(sv), axis=0)
            order = np.argsort(mean_abs)[::-1][:15]

            plt.barh(np.array(feature_names)[order[::-1]], mean_abs[order[::-1]])
            st.pyplot(plt.gcf())
            plt.clf()

            # Force plot per prediction
            st.write("Per-prediction explanation:")
            shap.initjs()
            st.write(
                shap.force_plot(
                    explainer.expected_value[1] if isinstance(shap_vals, list) else explainer.expected_value,
                    sv[0],
                    feature_names,
                    matplotlib=True
                )
            )

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
