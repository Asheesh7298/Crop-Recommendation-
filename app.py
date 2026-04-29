import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------
# Helper to evaluate models (used during training)
# ---------------------------------------------------
def evaluate_dl(model, X_test, y_test, label_encoder):
    y_true = np.argmax(y_test, axis=1)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc


# ---------------------------------------------------
# Training for BOTH datasets (cached)
# ---------------------------------------------------
@st.cache_resource
def load_and_train_all_datasets():
    datasets = {}

    # =========================
    # 1) Crop_recommendation.xlsx
    # =========================
    df1 = pd.read_excel("Crop_recommendation.xlsx")
    target_col1 = "label"

    X1 = df1.drop(columns=[target_col1])
    y1 = df1[target_col1]

    le1 = LabelEncoder()
    y1_enc = le1.fit_transform(y1)
    y1_ohe = tf.keras.utils.to_categorical(y1_enc)

    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1)
    feature_cols1 = X1.columns.tolist()
    cat_cols1 = []  # no categorical features

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1_scaled, y1_ohe, test_size=0.2, random_state=42, stratify=y1_enc
    )

    input_dim1 = X1_train.shape[1]
    num_classes1 = y1_train.shape[1]

    models1 = {}

    # ---- DNN / MLP ----
    mlp1 = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim1,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes1, activation='softmax')
    ])
    mlp1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    mlp1.fit(
        X1_train, y1_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[es1],
        verbose=0
    )
    models1["DNN / MLP"] = mlp1

    # ---- Autoencoder + DNN ----
    encoding_dim1 = 4
    inp1 = layers.Input(shape=(input_dim1,))
    enc1 = layers.Dense(32, activation='relu')(inp1)
    enc1 = layers.Dense(encoding_dim1, activation='relu')(enc1)
    dec1 = layers.Dense(32, activation='relu')(enc1)
    dec1 = layers.Dense(input_dim1, activation='linear')(dec1)

    autoencoder1 = models.Model(inp1, dec1)
    encoder1 = models.Model(inp1, enc1)
    autoencoder1.compile(optimizer='adam', loss='mse')

    ae_es1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    autoencoder1.fit(
        X1_train, X1_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[ae_es1],
        verbose=0
    )

    X1_train_enc = encoder1.predict(X1_train)
    X1_test_enc = encoder1.predict(X1_test)

    clf1 = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(encoding_dim1,)),
        layers.Dense(num_classes1, activation='softmax')
    ])
    clf1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    clf_es1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    clf1.fit(
        X1_train_enc, y1_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[clf_es1],
        verbose=0
    )
    models1["Autoencoder + DNN"] = {"encoder": encoder1, "clf": clf1}

    # ---- Keras Transformer ----
    X1_train_tr = X1_train.reshape(-1, input_dim1, 1)
    X1_test_tr = X1_test.reshape(-1, input_dim1, 1)

    d_model1 = 32
    num_heads1 = 4

    tinp1 = layers.Input(shape=(input_dim1, 1))
    x1 = layers.Dense(d_model1)(tinp1)
    attn1 = layers.MultiHeadAttention(num_heads=num_heads1, key_dim=d_model1)(x1, x1)
    x1 = layers.Add()([x1, attn1])
    x1 = layers.LayerNormalization()(x1)

    ffn1 = layers.Dense(64, activation='relu')(x1)
    ffn1 = layers.Dense(d_model1, activation='relu')(ffn1)
    x1 = layers.Add()([x1, ffn1])
    x1 = layers.LayerNormalization()(x1)

    x1 = layers.GlobalAveragePooling1D()(x1)
    x1 = layers.Dense(64, activation='relu')(x1)
    out1 = layers.Dense(num_classes1, activation='softmax')(x1)

    transformer1 = models.Model(tinp1, out1)
    transformer1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tr_es1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    transformer1.fit(
        X1_train_tr, y1_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[tr_es1],
        verbose=0
    )
    models1["Keras Transformer"] = transformer1

    # metadata for dataset 1
    datasets["Crop recommendation (NPK + weather)"] = {
        "models": models1,
        "scaler": scaler1,
        "label_encoder": le1,
        "feature_columns": feature_cols1,
        "cat_cols": cat_cols1,
        "raw_df": df1
    }

    # =========================
    # 2) sensor_Crop_Dataset (1).csv
    # =========================
    df2 = pd.read_csv("sensor_Crop_Dataset (1).csv")
    target_col2 = "Crop"

    X2 = df2.drop(columns=[target_col2])
    y2 = df2[target_col2]

    le2 = LabelEncoder()
    y2_enc = le2.fit_transform(y2)
    y2_ohe = tf.keras.utils.to_categorical(y2_enc)

    # Categorical columns in sensor dataset
    cat_cols2 = X2.select_dtypes(include=["object", "bool"]).columns.tolist()

    if len(cat_cols2) > 0:
        X2_proc = pd.get_dummies(X2, columns=cat_cols2)
    else:
        X2_proc = X2.copy()

    feature_cols2 = X2_proc.columns.tolist()

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2_proc)

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2_scaled, y2_ohe, test_size=0.2, random_state=42, stratify=y2_enc
    )

    input_dim2 = X2_train.shape[1]
    num_classes2 = y2_train.shape[1]

    models2 = {}

    # ---- DNN / MLP ----
    mlp2 = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim2,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes2, activation='softmax')
    ])
    mlp2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    mlp2.fit(
        X2_train, y2_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[es2],
        verbose=0
    )
    models2["DNN / MLP"] = mlp2

    # ---- Autoencoder + DNN ----
    encoding_dim2 = 4
    inp2 = layers.Input(shape=(input_dim2,))
    enc2 = layers.Dense(32, activation='relu')(inp2)
    enc2 = layers.Dense(encoding_dim2, activation='relu')(enc2)
    dec2 = layers.Dense(32, activation='relu')(enc2)
    dec2 = layers.Dense(input_dim2, activation='linear')(dec2)

    autoencoder2 = models.Model(inp2, dec2)
    encoder2 = models.Model(inp2, enc2)
    autoencoder2.compile(optimizer='adam', loss='mse')

    ae_es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    autoencoder2.fit(
        X2_train, X2_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[ae_es2],
        verbose=0
    )

    X2_train_enc = encoder2.predict(X2_train)
    X2_test_enc = encoder2.predict(X2_test)

    clf2 = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(encoding_dim2,)),
        layers.Dense(num_classes2, activation='softmax')
    ])
    clf2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    clf_es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    clf2.fit(
        X2_train_enc, y2_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[clf_es2],
        verbose=0
    )
    models2["Autoencoder + DNN"] = {"encoder": encoder2, "clf": clf2}

    # ---- Keras Transformer ----
    X2_train_tr = X2_train.reshape(-1, input_dim2, 1)
    X2_test_tr = X2_test.reshape(-1, input_dim2, 1)

    d_model2 = 32
    num_heads2 = 4

    tinp2 = layers.Input(shape=(input_dim2, 1))
    x2 = layers.Dense(d_model2)(tinp2)
    attn2 = layers.MultiHeadAttention(num_heads=num_heads2, key_dim=d_model2)(x2, x2)
    x2 = layers.Add()([x2, attn2])
    x2 = layers.LayerNormalization()(x2)

    ffn2 = layers.Dense(64, activation='relu')(x2)
    ffn2 = layers.Dense(d_model2, activation='relu')(ffn2)
    x2 = layers.Add()([x2, ffn2])
    x2 = layers.LayerNormalization()(x2)

    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dense(64, activation='relu')(x2)
    out2 = layers.Dense(num_classes2, activation='softmax')(x2)

    transformer2 = models.Model(tinp2, out2)
    transformer2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tr_es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    transformer2.fit(
        X2_train_tr, y2_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[tr_es2],
        verbose=0
    )
    models2["Keras Transformer"] = transformer2

    # store metadata (also keep soil type & variety choices for UI)
    soil_types = sorted(df2["Soil_Type"].dropna().unique().tolist())
    varieties = sorted(df2["Variety"].dropna().unique().tolist())

    datasets["Sensor crop dataset (sensors + soil + variety)"] = {
        "models": models2,
        "scaler": scaler2,
        "label_encoder": le2,
        "feature_columns": feature_cols2,
        "cat_cols": cat_cols2,
        "raw_df": df2,
        "soil_types": soil_types,
        "varieties": varieties
    }

    return datasets


# ---------------------------------------------------
# Helper: prepare one input row according to dataset metadata
# ---------------------------------------------------
def prepare_input_row(sample_dict, meta):
    df_sample = pd.DataFrame([sample_dict])

    cat_cols = meta["cat_cols"]
    feature_cols = meta["feature_columns"]
    scaler = meta["scaler"]

    if cat_cols:
        df_proc = pd.get_dummies(df_sample, columns=cat_cols)
    else:
        df_proc = df_sample.copy()

    # align columns with training
    df_proc = df_proc.reindex(columns=feature_cols, fill_value=0)

    X_scaled = scaler.transform(df_proc)
    return X_scaled


# ---------------------------------------------------
# Prediction helpers
# ---------------------------------------------------
def predict_with_mlp(meta, sample_dict):
    X_scaled = prepare_input_row(sample_dict, meta)
    model = meta["models"]["DNN / MLP"]
    le = meta["label_encoder"]
    preds = model.predict(X_scaled)
    idx = np.argmax(preds)
    return le.inverse_transform([idx])[0]


def predict_with_autoencoder(meta, sample_dict):
    X_scaled = prepare_input_row(sample_dict, meta)
    encoder = meta["models"]["Autoencoder + DNN"]["encoder"]
    clf = meta["models"]["Autoencoder + DNN"]["clf"]
    le = meta["label_encoder"]
    encoded = encoder.predict(X_scaled)
    preds = clf.predict(encoded)
    idx = np.argmax(preds)
    return le.inverse_transform([idx])[0]


def predict_with_transformer(meta, sample_dict):
    X_scaled = prepare_input_row(sample_dict, meta)
    model = meta["models"]["Keras Transformer"]
    le = meta["label_encoder"]
    input_dim = X_scaled.shape[1]
    X_tr = X_scaled.reshape(1, input_dim, 1)
    preds = model.predict(X_tr)
    idx = np.argmax(preds)
    return le.inverse_transform([idx])[0]


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="Crop Recommendation - Multi Dataset", layout="centered")
st.title("🌾 Crop Recommendation — 2 Datasets")

st.write("Select a dataset, choose an algorithm, enter feature values, and get a crop prediction.")

with st.spinner("Training models (only on first run)…"):
    datasets_meta = load_and_train_all_datasets()

dataset_name = st.selectbox("Choose dataset", list(datasets_meta.keys()))
meta = datasets_meta[dataset_name]

algo = st.selectbox("Choose Algorithm", ["DNN / MLP", "Autoencoder + DNN", "Keras Transformer"])

st.subheader("Input Features")

# ------- UI for each dataset -------
if "Crop recommendation" in dataset_name:
    # Dataset 1: original crop_recommendation
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)

    sample = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

else:
    # Dataset 2: sensor crop dataset
    col1, col2 = st.columns(2)

    with col1:
        Nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        Phosphorus = st.number_input("Phosphorus", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        Potassium = st.number_input("Potassium", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
        pH_Value = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        Temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
        Rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)

        soil_types = meta.get("soil_types", [])
        varieties = meta.get("varieties", [])
        Soil_Type = st.selectbox("Soil Type", soil_types if soil_types else ["Loamy"])
        Variety = st.selectbox("Variety", varieties if varieties else ["Default"])

    sample = {
        "Nitrogen": Nitrogen,
        "Phosphorus": Phosphorus,
        "Potassium": Potassium,
        "Temperature": Temperature,
        "Humidity": Humidity,
        "pH_Value": pH_Value,
        "Rainfall": Rainfall,
        "Soil_Type": Soil_Type,
        "Variety": Variety
    }

# ------- Predict button -------
if st.button("Predict Crop"):
    if algo == "DNN / MLP":
        pred = predict_with_mlp(meta, sample)
    elif algo == "Autoencoder + DNN":
        pred = predict_with_autoencoder(meta, sample)
    else:
        pred = predict_with_transformer(meta, sample)

    st.success(f"🌱 Recommended Crop: **{pred}**")
