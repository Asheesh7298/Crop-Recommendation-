import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------
# Helper to evaluate models
# ---------------------------------------------------
def evaluate_dl(model, X_test, y_test):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return accuracy_score(y_true, y_pred)

# ---------------------------------------------------
# Training (cached)
# ---------------------------------------------------
@st.cache_resource
def load_and_train_all_datasets():
    datasets = {}

    # =========================
    # DATASET 1
    # =========================
    df1 = pd.read_excel("Crop_recommendation.xlsx")
    X1 = df1.drop(columns=["label"])
    y1 = df1["label"]

    le1 = LabelEncoder()
    y1_enc = le1.fit_transform(y1)
    y1_ohe = tf.keras.utils.to_categorical(y1_enc)

    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1_scaled, y1_ohe, test_size=0.2, random_state=42, stratify=y1_enc
    )

    input_dim1 = X1_train.shape[1]
    num_classes1 = y1_train.shape[1]

    models1 = {}

    # ---- MLP ----
    mlp1 = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim1,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes1, activation='softmax')
    ])
    mlp1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mlp1.fit(X1_train, y1_train, epochs=20, batch_size=32, verbose=0)
    models1["DNN / MLP"] = mlp1
    mlp1_acc = evaluate_dl(mlp1, X1_test, y1_test)

    # ---- Autoencoder ----
    inp = layers.Input(shape=(input_dim1,))
    enc = layers.Dense(16, activation='relu')(inp)
    enc = layers.Dense(4, activation='relu')(enc)
    dec = layers.Dense(16, activation='relu')(enc)
    dec = layers.Dense(input_dim1)(dec)

    auto = models.Model(inp, dec)
    encoder = models.Model(inp, enc)
    auto.compile(optimizer='adam', loss='mse')
    auto.fit(X1_train, X1_train, epochs=20, verbose=0)

    X1_train_enc = encoder.predict(X1_train)
    X1_test_enc = encoder.predict(X1_test)

    clf = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(4,)),
        layers.Dense(num_classes1, activation='softmax')
    ])
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    clf.fit(X1_train_enc, y1_train, epochs=20, verbose=0)

    models1["Autoencoder + DNN"] = {"encoder": encoder, "clf": clf}
    ae_preds = clf.predict(X1_test_enc)
    ae_acc = accuracy_score(np.argmax(y1_test, axis=1), np.argmax(ae_preds, axis=1))

    # ---- Transformer ----
    X1_tr = X1_train.reshape(-1, input_dim1, 1)
    X1_test_tr = X1_test.reshape(-1, input_dim1, 1)

    inp = layers.Input(shape=(input_dim1, 1))
    x = layers.Dense(32)(inp)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(num_classes1, activation='softmax')(x)

    transformer = models.Model(inp, out)
    transformer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    transformer.fit(X1_tr, y1_train, epochs=20, verbose=0)

    models1["Keras Transformer"] = transformer
    tr_preds = transformer.predict(X1_test_tr)
    tr_acc = accuracy_score(np.argmax(y1_test, axis=1), np.argmax(tr_preds, axis=1))

    datasets["Crop recommendation"] = {
        "models": models1,
        "scaler": scaler1,
        "label_encoder": le1,
        "feature_columns": X1.columns.tolist(),
        "cat_cols": [],
        "accuracies": {
            "DNN / MLP": mlp1_acc,
            "Autoencoder + DNN": ae_acc,
            "Keras Transformer": tr_acc
        }
    }

    return datasets

# ---------------------------------------------------
# Prepare input
# ---------------------------------------------------
def prepare_input(sample, meta):
    df = pd.DataFrame([sample])
    df = df.reindex(columns=meta["feature_columns"], fill_value=0)
    return meta["scaler"].transform(df)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🌾 Crop Recommendation System")

data = load_and_train_all_datasets()
meta = data["Crop recommendation"]

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
    "N": N, "P": P, "K": K,
    "temperature": temp,
    "humidity": hum,
    "ph": ph,
    "rainfall": rain
}

if st.button("Predict"):
    X = prepare_input(sample, meta)

    if algo == "DNN / MLP":
        pred = meta["models"][algo].predict(X)
    elif algo == "Autoencoder + DNN":
        enc = meta["models"][algo]["encoder"].predict(X)
        pred = meta["models"][algo]["clf"].predict(enc)
    else:
        X = X.reshape(1, X.shape[1], 1)
        pred = meta["models"][algo].predict(X)

    result = meta["label_encoder"].inverse_transform([np.argmax(pred)])[0]
    st.success(f"🌱 Recommended Crop: {result}")

# Show comparison
st.subheader("📊 Model Comparison")
df_acc = pd.DataFrame({
    "Model": list(meta["accuracies"].keys()),
    "Accuracy (%)": [round(v*100, 2) for v in meta["accuracies"].values()]
})
st.table(df_acc)