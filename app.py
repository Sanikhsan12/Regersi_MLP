import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and scalers with custom objects for compatibility
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, R2Score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU

# A dictionary mapping custom objects to their respective classes/functions
custom_objects = {
    'mse': MeanSquaredError(),
    'r2_score': R2Score(),
    'LeakyReLU': LeakyReLU(),
    'l2': l2
}

try:
    model = load_model('mlp_model_v3.h5', custom_objects=custom_objects)
    x_scaler = joblib.load('x_scaler_v3.pkl')
    y_scaler = joblib.load('y_scaler_v3.pkl')
    print("Model and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    model = None
    x_scaler = None
    y_scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not x_scaler or not y_scaler:
        return jsonify({'error': 'Model atau scaler gagal dimuat. Silakan periksa file.'}), 500

    data = request.json
    try:
        # Get the input values from the request, ensuring all 10 are present
        input_data = [
            float(data['luas_tanah_m2']),
            float(data['luas_bangunan_m2']),
            float(data['jumlah_lantai']),
            float(data['jumlah_kamar_tidur']),
            float(data['jumlah_kamar_mandi']),
            float(data['luas_basement_m2']),
            float(data['kualitas_pemandangan']),
            float(data['pemandangan_air']),
            float(data['usia_rumah']),
            float(data['direnovasi'])
        ]
        
        # Reshape and scale the input data
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = x_scaler.transform(input_array)
        
        # Make a prediction
        scaled_prediction = model.predict(scaled_input)
        
        # Inverse transform the prediction to get the original scale
        prediction = y_scaler.inverse_transform(scaled_prediction)
        
        # Convert the numpy.float32 to a standard Python float before returning
        return jsonify({'prediction': float(prediction[0][0])})
    except KeyError as e:
        return jsonify({'error': f"Input tidak lengkap: {e} tidak ditemukan"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)