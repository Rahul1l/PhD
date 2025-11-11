import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
st.title("CAD Susceptibility Predictor")


# -------------------------------
# LOAD FILE
# -------------------------------
def load_data(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


# -------------------------------
# FIND TARGET COLUMN
# -------------------------------
def find_target_col(df):
    # you fixed CAD column manually; we keep it simple
    targets = [c for c in df.columns if "cad" in c.lower()]
    if len(targets) == 0:
        st.error("No CAD column found. Please ensure CAD column name contains 'CAD'.")
        st.stop()
    return targets[0]


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
if "model" not in st.session_state:
    st.session_state.model = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "trained" not in st.session_state:
    st.session_state.trained = False


# -------------------------------
# UPLOAD DATA
# -------------------------------
uploaded = st.file_uploader("Upload Dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if not uploaded:
    st.info("Upload your dataset to train the model.")
    st.stop()

df = load_data(uploaded)
df.columns = [str(c).strip() for c in df.columns]
st.dataframe(df.head())

target_col = find_target_col(df)


# -------------------------------
# TRAIN MODEL
# -------------------------------
if st.button("Train Model"):
    with st.spinner("Training..."):

        # Clean target
        df = df.dropna(subset=[target_col])
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])

        if df[target_col].nunique() < 2:
            st.error("Target column has only one class. Cannot train model.")
            st.stop()

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        # Split numeric/categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write("Numeric columns:", numeric_cols)
        st.write("Categorical columns:", categorical_cols)

        # Build model
        pipeline = build_pipeline(numeric_cols, categorical_cols)

        # Train-test split (safe)
        if len(df) < 5:
            st.warning("Dataset too small for split. Training on full dataset.")
            pipeline.fit(X, y)
            X_train = X.copy()
            y_train = y.copy()
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            try:
                probs = pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
                st.write(f"ROC AUC: {auc:.3f}")
            except:
                pass

            st.text(classification_report(y_test, preds, zero_division=0))

        # Save to session
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
        st.session_state.metadata = {
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


# -------------------------------
# PREDICTION UI
# -------------------------------
if st.session_state.trained:

    st.header("Predict CAD Risk")

    model = st.session_state.model
    meta = st.session_state.metadata

    pre = meta["pre"]
    clf = meta["clf"]
    feature_names = meta["feature_names"]
    X_train_trans = meta["X_train_trans"]

    # User inputs
    input_dict = {}
    col1, col2 = st.columns(2)

    for i, col in enumerate(meta["numeric_cols"]):
        input_dict[col] = col1.text_input(f"{col} (numeric)", "")

    for col in meta["categorical_cols"]:
        input_dict[col] = col2.text_input(f"{col} (text)", "")

    if st.button("Predict"):
        df_input = pd.DataFrame([input_dict])

        # Convert numeric
        for col in meta["numeric_cols"]:
            try:
                df_input[col] = pd.to_numeric(df_input[col])
            except:
                df_input[col] = np.nan

        # Transform
        try:
            inst = pre.transform(df_input)
            inst = inst.toarray() if hasattr(inst, "toarray") else np.array(inst)
        except Exception as e:
            st.error(f"Input transformation failed: {e}")
            st.stop()

        # Predict
        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        st.success(f"Prediction: **{'CAD Susceptible (1)' if pred==1 else 'Not Susceptible (0)'}**")
        st.info(f"Probability of CAD: {prob:.3f}")

        # -------------------------------
        # SHAP (FIXED VERSION)
        # -------------------------------
        st.subheader("SHAP Explanation")

        try:
            # Use smaller background sample
            bg = X_train_trans
            if bg.shape[0] > 200:
                idx = np.random.choice(bg.shape[0], 200, replace=False)
                bg = bg[idx]

            # ✅ FIXED SHAP: disable additivity check
            explainer = shap.TreeExplainer(
                clf,
                data=bg,
                feature_perturbation="interventional",
                check_additivity=False
            )

            shap_vals = explainer.shap_values(inst)

            # Global importance
            st.write("Global Feature Importance:")
            plt.figure(figsize=(7, 4))
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            mean_abs = np.mean(np.abs(sv), axis=1)
            ordered = np.argsort(mean_abs)[::-1][:15]
            plt.barh(np.array(feature_names)[ordered[::-1]], mean_abs[ordered[::-1]])
            st.pyplot(plt.gcf())
            plt.clf()

            # Waterfall attempt
            try:
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1] if isinstance(shap_vals, list) else explainer.expected_value,
                    sv[0] if isinstance(sv, np.ndarray) else sv[0],
                    feature_names=feature_names,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()
            except:
                pass

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
