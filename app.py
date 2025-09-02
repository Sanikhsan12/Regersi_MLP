import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
)

FEATURE_COLUMNS = [
    'luas_tanah_m2', 'luas_bangunan_m2', 'jumlah_lantai', 'jumlah_kamar_tidur',
    'jumlah_kamar_mandi','luas_basement_m2','kualitas_pemandangan','pemandangan_air',
    'usia_rumah','direnovasi'
]

TARGET_COLUMN = 'harga_IDR'

def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.exception(e)
        st.stop()

def load_scaler(pkl_path: str):
    if not os.path.exists(pkl_path):
        st.error(f"File scaler tidak ditemukan: {pkl_path}")
        st.stop()
    try:
        return joblib.load(pkl_path)
    except Exception as e:
        st.exception(e)
        st.stop()

def currency_idr(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return "-"

def estimate_interval(yhat: np.ndarray, y_scaler) -> tuple[float, float]:
    """Beri interval perkiraan sederhana ±7.5% dari prediksi (bukan CI statistik)."""
    y_hat = float(yhat)
    lower = y_hat * 0.925
    upper = y_hat * 1.075
    return lower, upper

st.title("🏠 Prediksi Harga Rumah (MLP)")
st.write(
    "Aplikasi ini digunakan untuk memprediksi harga rumah "
    "berdasarkan beberapa fitur kunci. Masukkan semua fitur di bawah ini untuk mendapatkan prediksi."
)

col_model, col_status = st.columns([3, 2])
with col_model:
    model = load_model("model.h5")
    x_scaler = load_scaler("x_scaler.pkl")
    y_scaler = load_scaler("y_scaler.pkl")

st.subheader("Input Spesifikasi Rumah")
with st.form("form_fitur", clear_on_submit=False, border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        luas_tanah_m2 = st.number_input("Luas tanah (m²)", min_value=0, value=0, step=10)
        jumlah_lantai = st.number_input("Jumlah lantai", min_value=0, value=0, step=1)
        jumlah_kamar_mandi = st.number_input("Jumlah kamar mandi", min_value=0, value=0, step=1)
        usia_rumah = st.number_input("Usia rumah (tahun)", min_value=0, value=0, step=1)
    with c2:
        luas_bangunan_m2 = st.number_input("Luas bangunan (m²)", min_value=0, value=0, step=10)
        jumlah_kamar_tidur = st.number_input("Jumlah kamar tidur", min_value=0, value=0, step=1)
        luas_basement_m2 = st.number_input("Luas basement (m²)", min_value=0, value=0, step=10)

        direnovasi = st.radio(
            "Pernah direnovasi?",
            options=[0, 1],
            index=0,
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True,
        )

    with c3:
        kualitas_pemandangan = st.radio(
            "Kualitas pemandangan",
            options=[1, 2, 3, 4, 5],
            index=2,
            format_func=lambda x: {1: "Sangat Buruk", 2: "Buruk", 3: "Cukup", 4: "Baik", 5: "Sangat Baik"}[x],
            horizontal=True,
        )

        pemandangan_air = st.radio(
            "Pemandangan air?",
            options=[0, 1],
            index=0,
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            horizontal=True,
        )

        st.markdown("\n")
        submitted = st.form_submit_button("Prediksi Harga", use_container_width=True)

input_row = pd.DataFrame([{
    'luas_tanah_m2': luas_tanah_m2,
    'luas_bangunan_m2': luas_bangunan_m2,
    'jumlah_lantai': jumlah_lantai,
    'jumlah_kamar_tidur': jumlah_kamar_tidur,
    'jumlah_kamar_mandi': jumlah_kamar_mandi,
    'luas_basement_m2': luas_basement_m2,
    'kualitas_pemandangan': kualitas_pemandangan,
    'pemandangan_air': pemandangan_air,
    'usia_rumah': usia_rumah,
    'direnovasi': direnovasi,
}])

# Prediksi Single Input
if submitted:
    # Validasi kolom
    missing_cols = [c for c in FEATURE_COLUMNS if c not in input_row.columns]
    if missing_cols:
        st.error(f"Kolom input tidak lengkap: {missing_cols}")
    else:
        try:
            x_scaled = x_scaler.transform(input_row[FEATURE_COLUMNS])
            y_scaled_pred = model.predict(x_scaled, verbose=0)
            y_pred = y_scaler.inverse_transform(y_scaled_pred)[0, 0]
            low, up = estimate_interval(y_pred, y_scaler)

            st.success("Prediksi berhasil!")
            metr1, metr2, metr3 = st.columns(3)
            with metr1:
                st.metric(label="Harga prediksi", value=currency_idr(y_pred))
            with metr2:
                st.metric(label="Perkiraan bawah (±7.5%)", value=currency_idr(low))
            with metr3:
                st.metric(label="Perkiraan atas (±7.5%)", value=currency_idr(up))
        except Exception as e:
            st.exception(e)