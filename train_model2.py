import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# * Membaca File CSV
data = pd.read_csv('data_final_cleaned.csv')

# * Inisialisasai target dan fitur
fitur_input = ['luas_tanah_m2', 'luas_bangunan_m2', 'jumlah_lantai', 'jumlah_kamar_tidur', 'jumlah_kamar_mandi','luas_basement_m2','kualitas_pemandangan','pemandangan_air','usia_rumah','direnovasi']
x = data[fitur_input]
y = data[['harga_IDR']]

# * Normalisasi Data
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# * Skalalisasi Rentang
x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)

# * Modelisasi
fungsi_aktivasi = 'leaky_relu'
lambda_l2 = 0.00002
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(fitur_input),), name='input_layer'),
    tf.keras.layers.Dense(units=256, activation=fungsi_aktivasi, name='hidden_layer_1', kernel_regularizer=l2(lambda_l2)),
    tf.keras.layers.Dropout(0.015, name='dropout_layer_1'),
    tf.keras.layers.Dense(units=128, activation=fungsi_aktivasi, name='hidden_layer_2', kernel_regularizer=l2(lambda_l2)),
    tf.keras.layers.Dropout(0.015, name='dropout_layer_2'),
    tf.keras.layers.Dense(units=64, activation=fungsi_aktivasi, name='hidden_layer_3', kernel_regularizer=l2(lambda_l2)),
    tf.keras.layers.Dropout(0.015, name='dropout_layer_3'),
    tf.keras.layers.Dense(units=32, activation=fungsi_aktivasi, name='hidden_layer_4', kernel_regularizer=l2(lambda_l2)),
    tf.keras.layers.Dropout(0.015, name='dropout_layer_4'),
    tf.keras.layers.Dense(units=1, name='output_layer')
])

# * Visualisasi model
model.summary()

# * Kompilasi Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=[tf.keras.metrics.R2Score()]
)

# * Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

# * Train Model
history = model.fit(x_scaled, y_scaled, epochs=2000, batch_size=32, validation_split=0.3, verbose=1, callbacks=[early_stopping])

# * Evaluasi Dan Prediksi
r_squared_history = history.history['r2_score']

# * Menyimpan Scaller
joblib.dump(x_scaler, 'x_scaler_v3.pkl')
joblib.dump(y_scaler, 'y_scaler_v3.pkl')

# * Simpan model ke file untuk digunakan Flask app
model.save('mlp_model_v3.h5')

# * Plotting R-squared dan Epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(r_squared_history) + 1), r_squared_history)
plt.title('Perkembangan R-Squared Selama Training')
plt.xlabel('Epoch')
plt.ylabel('R-Squared ($R^2$)')
plt.grid(True)
plt.ylim(0, 1.1)
plt.show()

# * Plotting Loss Train dan Loss Validation
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Perkembangan Loss Selama Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# ! Prediksi data baru dengan semua fitur
data_baru = np.array([[300, 250, 2, 5, 4, 80, 4, 1, 15, 1]]) 
data_baru_scaled = x_scaler.transform(data_baru)
prediksi_scaled = model.predict(data_baru_scaled)
prediksi_harga = y_scaler.inverse_transform(prediksi_scaled)

print(f"\nPrediksi harga untuk rumah dengan fitur berikut:")
print(f"  - Luas tanah: {data_baru[0][0]} m²")
print(f"  - Luas bangunan: {data_baru[0][1]} m²")
print(f"  - Jumlah lantai: {data_baru[0][2]}")
print(f"  - Jumlah kamar tidur: {data_baru[0][3]}")
print(f"  - Jumlah kamar mandi: {data_baru[0][4]}")
print(f"  - Luas basement: {data_baru[0][5]} m²")
print(f"  - Kualitas pemandangan: {data_baru[0][6]}")
print(f"  - Pemandangan air: {'Ya' if data_baru[0][7] == 1 else 'Tidak'}")
print(f"  - Usia rumah: {data_baru[0][8]} tahun")
print(f"  - Direnovasi: {'Ya' if data_baru[0][9] == 1 else 'Tidak'}")
print(f"adalah: {prediksi_harga[0][0]:.2f} Rupiah")