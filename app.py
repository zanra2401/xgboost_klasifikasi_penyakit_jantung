import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT MODEL ---
# Menggunakan cache agar model hanya dimuat sekali
@st.cache_resource
def load_model(path):
    """Memuat model dari file .pkl"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {path}")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

# Memuat model yang sudah dilatih
model_path = "klasifikasi_penyakit_jantung_xgboost.pkl"
model = load_model(model_path)

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("❤️ Aplikasi Prediksi Penyakit Jantung")
st.write("""
Aplikasi ini menggunakan model Machine Learning (XGBoost) untuk memprediksi risiko penyakit jantung berdasarkan data medis pasien. 
Silakan masukkan data pasien pada panel di sebelah kiri.
""")

# Sidebar untuk input dari pengguna
st.sidebar.header("Input Data Pasien")

def user_input_features():
    """Mengambil input dari pengguna melalui sidebar"""
    st.sidebar.markdown("**Fitur Biometrik**")
    age = st.sidebar.slider('Usia (Tahun)', 20, 80, 50)
    sex_options = {'Pria': 'Male', 'Wanita': 'Female'}
    sex = st.sidebar.radio('Jenis Kelamin', list(sex_options.keys()))

    st.sidebar.markdown("**Fitur Medis**")
    cp_options = {
        'Typical Angina': 'typical angina',
        'Atypical Angina': 'atypical angina',
        'Non-Anginal': 'non-anginal',
        'Asymptomatic': 'asymptomatic'
    }
    cp = st.sidebar.selectbox('Tipe Nyeri Dada (Chest Pain)', list(cp_options.keys()))

    trestbps = st.sidebar.slider('Tekanan Darah Istirahat (mm Hg)', 90, 200, 120)
    
    chol = st.sidebar.number_input('Kolesterol Serum (mg/dl)', min_value=100, max_value=600, value=200)

    fbs_options = {'Ya (> 120 mg/dl)': True, 'Tidak (<= 120 mg/dl)': False}
    fbs = st.sidebar.radio('Gula Darah Puasa > 120 mg/dl', list(fbs_options.keys()))

    restecg_options = {
        'Normal': 'normal',
        'LV Hypertrophy': 'lv hypertrophy',
        'ST-T Abnormality': 'st-t abnormality'
    }
    restecg = st.sidebar.selectbox('Hasil Elektrokardiogram Istirahat', list(restecg_options.keys()))

    thalach = st.sidebar.slider('Detak Jantung Maksimum Tercapai', 70, 220, 150)

    exang_options = {'Ya': True, 'Tidak': False}
    exang = st.sidebar.radio('Angina Akibat Olahraga (Exercise Induced Angina)', list(exang_options.keys()))

    # --- PERUBAHAN DI SINI ---
    # Mengubah slider menjadi number_input untuk kemudahan dan presisi
    oldpeak = st.sidebar.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=7.0, value=1.0, step=0.1)

    slope_options = {
        'Upsloping': 'upsloping',
        'Flat': 'flat',
        'Downsloping': 'downsloping'
    }
    slope = st.sidebar.selectbox('Slope dari Puncak Latihan ST Segment', list(slope_options.keys()))
    
    ca = st.sidebar.slider('Jumlah Pembuluh Darah Utama (0-3)', 0, 3, 0)
    
    thal_options = {
        'Normal': 'normal',
        'Fixed Defect': 'fixed defect',
        'Reversable Defect': 'reversable defect'
    }
    thal = st.sidebar.selectbox('Thalassemia', list(thal_options.keys()))

    # Mengumpulkan data dalam dictionary
    data = {
        'age': age,
        'sex': sex_options[sex],
        'cp': cp_options[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_options[fbs],
        'restecg': restecg_options[restecg],
        'thalach': thalach,
        'exang': exang_options[exang],
        'oldpeak': oldpeak,
        'slope': slope_options[slope],
        'ca': float(ca),
        'thal': thal_options[thal]
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Mengambil input pengguna
input_df = user_input_features()

# Menampilkan data input mentah
st.subheader("Data Pasien yang Dimasukkan")
st.write(input_df)

# Tombol untuk melakukan prediksi
predict_button = st.button("Lakukan Prediksi", type="primary")

# --- LOGIKA PREDIKSI ---
if predict_button and model is not None:
    # 1. Preprocessing data input agar sesuai dengan data training
    #    a. Lakukan One-Hot Encoding
    input_df_encoded = pd.get_dummies(input_df, dtype=int)
    
    #    b. Ambil daftar kolom yang diharapkan oleh model
    model_columns = model.get_booster().feature_names
    
    #    c. Sesuaikan kolom input dengan kolom model
    #       (tambah kolom yang hilang dengan nilai 0)
    final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 2. Lakukan Prediksi
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    
    # 3. Tampilkan Hasil
    st.subheader("Hasil Prediksi")
    
    # Asumsi: 0 = Tidak Sakit Jantung, 1 = Sakit Jantung
    if prediction[0] == 1:
        st.error("**Risiko Tinggi: Terdeteksi Penyakit Jantung**", icon="☠️")
        st.error(f"Pasien Terklasifikasi Ke Kelas: **{prediction[0]}**")
    else:
        st.success("**Risiko Rendah: Tidak Terdeteksi Penyakit Jantung**", icon="❤️")
        st.error(f"Pasien Terklasifikasi Ke Kelas: **{prediction[0]}**")

        
    # Tampilkan probabilitas
    st.write("Probabilitas Prediksi:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Probabilitas **Tidak** Sakit Jantung", value=f"{prediction_proba[0][0]*100:.2f}%")
    with col2:
        st.metric(label="Probabilitas Sakit Jantung", value=f"{prediction_proba[0][1]*100:.2f}%")

    # Disclaimer
    st.warning("""
    **Disclaimer:** Hasil prediksi ini berdasarkan model machine learning dan tidak boleh dianggap sebagai diagnosis medis final. 
    Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk diagnosis dan penanganan lebih lanjut.
    """)
elif predict_button and model is None:
    st.error("Prediksi tidak dapat dilakukan karena model gagal dimuat.")
