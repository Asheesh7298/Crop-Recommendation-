import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# ---------------------------------------------------
# Training (cached)
# ---------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_excel("Crop_recommendation.xlsx")

    X = df.drop(columns=["label"])
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {}
    accuracies = {}

    # -----------------------------
    # 1. Random Forest
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    models["Random Forest"] = rf
    accuracies["Random Forest"] = rf_acc

    # -----------------------------
    # 2. XGBoost
    # -----------------------------
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    models["XGBoost"] = xgb
    accuracies["XGBoost"] = xgb_acc

    # -----------------------------
    # 3. Logistic Regression
    # -----------------------------
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    models["Logistic Regression"] = lr
    accuracies["Logistic Regression"] = lr_acc

    return {
        "models": models,
        "accuracies": accuracies,
        "scaler": scaler,
        "label_encoder": le,
        "features": X.columns.tolist()
    }

# ---------------------------------------------------
# Prepare input
# ---------------------------------------------------
def prepare_input(sample, meta):
    df = pd.DataFrame([sample])
    df = df.reindex(columns=meta["features"], fill_value=0)
    return meta["scaler"].transform(df)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🌾 Crop Recommendation")

meta = load_and_train()

algo = st.selectbox("Choose Model", list(meta["models"].keys()))

# Show accuracy
acc = meta["accuracies"][algo]
st.info(f"📊 Accuracy: {acc*100:.2f}%")

# Inputs
N = st.number_input("Nitrogen", 0.0, 200.0, 50.0)
P = st.number_input("Phosphorus", 0.0, 200.0, 40.0)
K = st.number_input("Potassium", 0.0, 200.0, 40.0)
temp = st.number_input("Temperature", 0.0, 60.0, 25.0)
hum = st.number_input("Humidity", 0.0, 100.0, 80.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rain = st.number_input("Rainfall", 0.0, 500.0, 150.0)

sample = {
    "N": N,
    "P": P,
    "K": K,
    "temperature": temp,
    "humidity": hum,
    "ph": ph,
    "rainfall": rain
}

# Prediction
if st.button("Predict"):
    X = prepare_input(sample, meta)
    model = meta["models"][algo]

    pred = model.predict(X)
    result = meta["label_encoder"].inverse_transform(pred)[0]

    st.success(f"🌱 Recommended Crop: {result}")

# Comparison Table
st.subheader("📊 Model Comparison")

df_acc = pd.DataFrame({
    "Model": list(meta["accuracies"].keys()),
    "Accuracy (%)": [round(v*100, 2) for v in meta["accuracies"].values()]
})

st.table(df_acc)