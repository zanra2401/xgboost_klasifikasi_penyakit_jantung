import streamlit as st
import pandas as pd
import numpy as np # Import numpy jika Anda menggunakan np.array() atau sejenisnya
import pickle

st.set_page_config(page_title="Prediksi Risiko Penyakit Jantung", layout="centered")
    

# --- Muat Model yang Sudah Dilatih ---
# Pastikan file model_klasifikasi_xgboost.pkl ada di direktori yang sama
# atau berikan path lengkap ke file tersebut.
try:
    with open('klasifikasi_penyakit_jantung_xgboost.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: File model 'model_klasifikasi_xgboost.pkl' tidak ditemukan.")
    st.info("Pastikan Anda sudah melatih model XGBoost dan menyimpannya dengan nama tersebut.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

# --- JIKA ANDA MENGGUNAKAN SCALER/ENCODER SAAT PELATIHAN, MUAT JUGA DI SINI ---
# Contoh:
# try:
#     scaler = joblib.load('scaler_data.pkl') # Ganti dengan nama file scaler Anda
#     st.success("Scaler berhasil dimuat!")
# except FileNotFoundError:
#     st.warning("Peringatan: File scaler tidak ditemukan. Jika model Anda dilatih dengan data yang diskalakan, prediksi mungkin tidak akurat.")
#     scaler = None # Atur None jika tidak ada scaler

# --- Judul Aplikasi Streamlit ---
st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.markdown("Aplikasi ini memprediksi risiko penyakit jantung berdasarkan input fitur pasien.")

st.write("---")

# --- Form Input untuk Fitur Pasien ---
st.header("Input Data Pasien")

# Mengumpulkan input dari pengguna
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Usia", 1, 120, 50)
    sex = st.selectbox("Jenis Kelamin", options=[(0, "Perempuan"), (1, "Laki-laki")], format_func=lambda x: x[1])[0] # 0: Perempuan, 1: Laki-laki
    cp = st.selectbox("Tipe Nyeri Dada (cp)", options=[0, 1, 2, 3], help="0: Asymptomatic, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Typical Angina")
    trestbps = st.number_input("Tekanan Darah Istirahat (trestbps)", 80, 200, 120, help="Tekanan darah sistolik saat istirahat (mm Hg)")

with col2:
    chol = st.number_input("Kolesterol Serum (chol)", 100, 600, 200, help="Kolesterol serum dalam mg/dl")
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0] # 0: False, 1: True
    restecg = st.selectbox("Hasil Elektrokardiogram Saat Istirahat (restecg)", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
    thalach = st.number_input("Detak Jantung Maksimal Tercapai (thalach)", 70, 220, 150, help="Detak jantung maksimal yang tercapai saat uji stres")

with col3:
    exang = st.selectbox("Angina Akibat Olahraga (exang)", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0] # 0: No, 1: Yes
    oldpeak = st.number_input("Depresi ST Akibat Olahraga Relatif Terhadap Istirahat (oldpeak)", 0.0, 6.0, 1.0, step=0.1, help="Penurunan ST yang diinduksi olahraga relatif terhadap istirahat")
    slope = st.selectbox("Kemiringan Segmen ST Puncak Latihan (slope)", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    ca = st.selectbox("Jumlah Pembuluh Darah Besar (ca)", options=[0, 1, 2, 3, 4], help="Jumlah pembuluh darah besar (0-3) yang diwarnai oleh fluoroskopi")
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed defect, 2: Reversable defect, 3: Other types (rarely used)") # Biasanya 1,2,3, tapi contoh data ada 0


# --- Tombol Prediksi ---
if st.button("Prediksi Risiko Penyakit Jantung"):
    # Kumpulkan semua input ke dalam dictionary
    input_data = {
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    }

    # Buat DataFrame dari input
    # Penting: Pastikan urutan kolom sesuai dengan saat model dilatih!
    # Urutan ini harus sama dengan X_train Anda.
    input_df = pd.DataFrame(input_data)

    # --- LAKUKAN PRA-PEMROSESAN YANG SAMA DENGAN DATA PELATIHAN ---
    # Jika Anda menggunakan scaler atau encoder, terapkan di sini.
    # Contoh jika Anda menggunakan StandardScaler:
    # if scaler:
    #     input_processed = scaler.transform(input_df)
    # else:
    #     input_processed = input_df
    
    input_processed = input_df # Untuk contoh ini, asumsi tidak ada pra-pemrosesan kompleks

    # --- Buat Prediksi ---
    prediction = model.predict(input_processed)
    prediction_proba = model.predict_proba(input_processed)

    st.write("---")
    st.header("Hasil Prediksi")

    if prediction[0] == 1:
        st.error("⚠️ Pasien Diprediksi **BERISIKO TINGGI** Terkena Penyakit Jantung")
        st.write(f"Terklasifikasi pada kelas: **{prediction[0]}**")
        st.write(f"Probabilitas risiko tidak terkena penyakit jantung: **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Probabilitas risiko terkena penyakit jantung: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.write(f"Terklasifikasi pada kelas: **{prediction[0]}**")
        st.success("✅ Pasien Diprediksi **BERISIKO RENDAH** Terkena Penyakit Jantung")
        st.write(f"Probabilitas risiko tidak terkena penyakit jantung: **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Probabilitas risiko terkena penyakit jantung: **{prediction_proba[0][1]*100:.2f}%**")
