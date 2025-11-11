# app.py
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
import io

st.set_page_config(page_title="CAD Susceptibility App", layout="wide")
st.title("🩺 CAD Susceptibility Predictor (Upload-only)")

# ---------------------
# Helpers
# ---------------------
def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def find_target_col(df):
    # exact column from your sheet
    candidates = ["CAD (0 & 1 is normal)", "CAD (0 & 1 is normal) "]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try common names
    for c in df.columns:
        if str(c).strip().lower() in ("cad","target","disease","label","outcome"):
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
    ], remainder="drop")
    pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    return pipe

# ---------------------
# UI - upload & train
# ---------------------
uploaded_file = st.file_uploader("Upload dataset (.csv or .xlsx) — required", type=["csv", "xlsx"])

# store in session
if "model_obj" not in st.session_state:
    st.session_state.model_obj = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        # strip column names
        df.columns = [str(c).strip() for c in df.columns]
        st.session_state.df = df
        st.success(f"Loaded dataset — shape: {df.shape}")
        st.dataframe(df.head(8))
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.info("Please upload your dataset. The app will train from the uploaded file.")
    st.stop()

# Train button
if st.button("Train model from uploaded dataset"):
    with st.spinner("Training..."):
        df = st.session_state.df.copy()

        # target col
        target_col = find_target_col(df)
        st.write(f"Using target column: **{target_col}**")
        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        if df.shape[0] < 10:
            st.warning("After dropping missing targets there are < 10 rows — model may not be reliable.")

        # Ensure binary integer target (0/1). attempt conversion
        try:
            df[target_col] = pd.to_numeric(df[target_col])
            # If values are not 0/1 but numeric, try to coerce: treat >1 as 1 (but we assume dataset binary)
            uniq = sorted(df[target_col].dropna().unique())
            if not set(uniq).issubset({0,1}):
                # if values are like "0", "1" as strings it's fine; otherwise map nonzero to 1
                st.info(f"Detected target values: {uniq}. Will coerce non-zero to 1.")
                df[target_col] = df[target_col].apply(lambda x: 0 if float(x) == 0 else 1)
            df[target_col] = df[target_col].astype(int)
        except Exception:
            # fallback: map strings
            df[target_col] = df[target_col].astype(str).str.strip().str.lower().map({
                "0":0, "1":1, "no":0, "normal":0, "yes":1, "positive":1, "disease":1
            })
            if df[target_col].isna().any():
                st.error("Could not convert some target values to binary (0/1). Please clean the target column.")
                st.stop()

        # Drop ID column if present
        cols_to_drop = []
        if "Patient ID" in df.columns:
            cols_to_drop.append("Patient ID")
        X = df.drop(columns=cols_to_drop + [target_col])
        y = df[target_col]

        # Keep Glucose as categorical (per your choice)
        # Identify numeric/categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        st.write("Detected numeric columns:", numeric_cols)
        st.write("Detected categorical columns:", categorical_cols)

        # Build and train pipeline
        pipeline = build_pipeline(numeric_cols, categorical_cols)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        except Exception:
            # maybe stratify fails due to single class — fallback without stratify
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.warning("Stratified split failed (single class?). Used random split instead.")

        pipeline.fit(X_train, y_train)

        # metrics
        preds = pipeline.predict(X_test)
        try:
            probs = pipeline.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = None

        st.write("### Validation report")
        st.text(classification_report(y_test, preds, zero_division=0))
        if auc is not None:
            st.write(f"ROC AUC (val): {auc:.3f}")

        # Save model object and metadata to session
        # For SHAP: capture preprocessor and classifier + a transformed sample
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        X_train_trans = pre.transform(X_train)
        try:
            X_train_trans = X_train_trans.toarray()
        except Exception:
            X_train_trans = np.array(X_train_trans)

        feature_names = None
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            # fallback
            feature_names = [f"f{i}" for i in range(X_train_trans.shape[1])]

        st.session_state.model_obj = {
            "pipeline": pipeline,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "target_col": target_col,
            "X_train_raw": X_train,
            "X_train_trans": X_train_trans,
            "feature_names": feature_names
        }
        st.session_state.trained = True
        st.success("Training complete and model saved in session.")

# ---------------------
# If trained -> predict UI
# ---------------------
if st.session_state.trained:
    obj = st.session_state.model_obj
    pipeline = obj["pipeline"]
    numeric_cols = obj["numeric_cols"]
    categorical_cols = obj["categorical_cols"]
    target_col = obj["target_col"]
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    st.markdown("---")
    st.header("Make a prediction")

    # Option: choose a row from uploaded df
    if st.checkbox("Fill inputs from a row in uploaded dataset"):
        idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(st.session_state.df)-1, value=0, step=1)
        if st.button("Load row"):
            row = st.session_state.df.iloc[int(idx)]
            values = {}
            for c in numeric_cols + categorical_cols:
                values[c] = row[c] if c in row.index else ""
            st.session_state.prefill = values
            st.success(f"Loaded row {idx}")

    prefill = st.session_state.get("prefill", {})

    # Input widgets
    st.write("Enter patient features (leave blank if unknown):")
    input_vals = {}
    cols_ui = st.columns(2)
    for i, c in enumerate(numeric_cols):
        default = str(prefill.get(c, ""))
        v = cols_ui[i % 2].text_input(c, value=default, key=f"num_{c}")
        input_vals[c] = v
    for c in categorical_cols:
        default = str(prefill.get(c, ""))
        input_vals[c] = st.text_input(c, value=default, key=f"cat_{c}")

    if st.button("Predict on input"):
        # build df
        df_input = pd.DataFrame([input_vals])
        # convert numeric columns
        for c in numeric_cols:
            if c in df_input.columns:
                try:
                    df_input[c] = pd.to_numeric(df_input[c])
                except Exception:
                    df_input[c] = np.nan
        for c in categorical_cols:
            if c not in df_input.columns:
                df_input[c] = np.nan

        try:
            pred = pipeline.predict(df_input)[0]
            prob = pipeline.predict_proba(df_input)[0][1] if hasattr(pipeline, "predict_proba") else None
            label = "Susceptible (CAD=1)" if int(pred) == 1 else "Not Susceptible (CAD=0)"
            st.success(f"Prediction: **{label}**")
            if prob is not None:
                st.info(f"Predicted probability of CAD: {prob:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        # SHAP explanations
        st.subheader("Explain prediction (SHAP)")
        try:
            # Prepare background (sample)
            X_bg = obj["X_train_trans"]
            if X_bg.shape[0] > 200:
                idxs = np.random.choice(X_bg.shape[0], 200, replace=False)
                X_bg_sample = X_bg[idxs]
            else:
                X_bg_sample = X_bg

            explainer = shap.TreeExplainer(clf, data=X_bg_sample, feature_perturbation="interventional")
            # transform df_input
            inst = pre.transform(df_input)
            try:
                inst = inst.toarray()
            except Exception:
                inst = np.array(inst)
            shap_vals = explainer.shap_values(inst)

            # Global importance (summary bar)
            st.write("Global feature importance (approx., from training sample)")
            plt.figure(figsize=(8,4))
            if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                # choose class 1 importance
                sv = shap_vals[1]
            else:
                sv = shap_vals
            # mean abs
            mean_abs = np.mean(np.abs(sv), axis=0)
            names = obj.get("feature_names", [f"f{i}" for i in range(len(mean_abs))])
            order = np.argsort(mean_abs)[::-1][:20]
            plt.barh(np.array(names)[order[::-1]], mean_abs[order[::-1]])
            plt.title("Mean |SHAP value| (top features)")
            st.pyplot(plt.gcf())
            plt.clf()

            # Per-instance waterfall / contributions
            st.write("Per-prediction feature contributions")
            if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                inst_shap = shap_vals[1][0]
                base = explainer.expected_value[1]
            else:
                inst_shap = shap_vals[0][0]
                base = explainer.expected_value
            # waterfall plot
            try:
                shap.plots._waterfall.waterfall_legacy(base, inst_shap, feature_names=names, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception:
                # fallback bar of top signed contributions
                order2 = np.argsort(np.abs(inst_shap))[::-1][:15]
                plt.barh(np.array(names)[order2[::-1]], inst_shap[order2[::-1]])
                plt.title("Signed SHAP contributions (top features)")
                st.pyplot(plt.gcf())
                plt.clf()

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

else:
    st.info("Train a model first using your uploaded dataset.")
