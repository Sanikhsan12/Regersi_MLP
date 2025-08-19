from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load data and scaler (fitur harus sama dengan training)
data = pd.read_csv('data_final.csv')
fitur_input = ['luas_tanah_m2', 'luas_bangunan_m2', 'jumlah_lantai', 'jumlah_kamar_tidur', 'jumlah_kamar_mandi','luas_basement_m2','kualitas_pemandangan','pemandangan_air','usia_rumah','direnovasi']
x = data[fitur_input]
y = data[['harga_IDR']]

# Load scaler (re-fit to data, or load from file if saved)
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_scaler.fit(x)
y_scaler.fit(y)

# Load trained model
model = tf.keras.models.load_model('mlp_model.h5', compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data_baru = [
            float(request.form.get('luas_tanah_m2', 0)),
            float(request.form.get('luas_bangunan_m2', 0)),
            int(request.form.get('jumlah_lantai', 1)),
            int(request.form.get('jumlah_kamar_tidur', 1)),
            int(request.form.get('jumlah_kamar_mandi', 1)),
            float(request.form.get('luas_basement_m2', 0)),
            int(request.form.get('kualitas_pemandangan', 1)),
            int(request.form.get('pemandangan_air', 0)),
            int(request.form.get('usia_rumah', 0)),
            int(request.form.get('direnovasi', 0)),
        ]
        data_baru_np = np.array([data_baru])
        data_baru_scaled = x_scaler.transform(data_baru_np)
        prediksi_scaled = model.predict(data_baru_scaled)
        prediksi_harga = y_scaler.inverse_transform(prediksi_scaled)
        harga = float(prediksi_harga[0][0])
        return render_template('index.html', prediction_text=f'Prediksi harga rumah: {harga:,.2f} Rupiah')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
