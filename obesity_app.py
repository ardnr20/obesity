import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Judul Aplikasi
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi Machine Learning untuk memprediksi tingkat obesitas berdasarkan data gaya hidup")

# Load Data
data = pd.read_csv("ObesityDataSet.csv")

# Bersihkan nilai '?'
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Tampilkan data awal
st.subheader("Tampilan Data Awal")
st.dataframe(data.head())

# Drop baris yang masih mengandung NaN
data.dropna(inplace=True)

# Encode target terlebih dahulu
le_target = LabelEncoder()
data["NObeyesdad_encoded"] = le_target.fit_transform(data["NObeyesdad"])

# Encode fitur kategorikal
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    if col != "NObeyesdad":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Fitur dan target
X = data.drop(columns=["NObeyesdad", "NObeyesdad_encoded"])
y = data["NObeyesdad_encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Akurasi Model: {acc:.2f}")

# Form input pengguna
st.subheader("Form Input Prediksi")
gender = st.selectbox("Jenis Kelamin", label_encoders['Gender'].classes_)
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0)
family_history = st.selectbox("Riwayat Keluarga Overweight", label_encoders['family_history_with_overweight'].classes_)
FAVC = st.selectbox("Sering makan makanan berkalori tinggi?", label_encoders['FAVC'].classes_)
FCVC = st.slider("Frekuensi konsumsi sayur (1-3)", 1.0, 3.0, 2.0)
NCP = st.slider("Jumlah makan utama per hari", 1.0, 4.0, 3.0)
CAEC = st.selectbox("Makan di luar?", label_encoders['CAEC'].classes_)
SMOKE = st.selectbox("Merokok?", label_encoders['SMOKE'].classes_)
CH2O = st.slider("Konsumsi air (liter/hari)", 1.0, 3.0, 2.0)
SCC = st.selectbox("Monitor kalori yang dikonsumsi?", label_encoders['SCC'].classes_)
FAF = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 10.0, 2.0)
TUE = st.slider("Waktu layar (jam/hari)", 0.0, 10.0, 3.0)
CALC = st.selectbox("Konsumsi alkohol", label_encoders['CALC'].classes_)
MTRANS = st.selectbox("Transportasi", label_encoders['MTRANS'].classes_)

# Encode input
input_dict = {
    "Gender": label_encoders['Gender'].transform([gender])[0],
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history_with_overweight": label_encoders['family_history_with_overweight'].transform([family_history])[0],
    "FAVC": label_encoders['FAVC'].transform([FAVC])[0],
    "FCVC": FCVC,
    "NCP": NCP,
    "CAEC": label_encoders['CAEC'].transform([CAEC])[0],
    "SMOKE": label_encoders['SMOKE'].transform([SMOKE])[0],
    "CH2O": CH2O,
    "SCC": label_encoders['SCC'].transform([SCC])[0],
    "FAF": FAF,
    "TUE": TUE,
    "CALC": label_encoders['CALC'].transform([CALC])[0],
    "MTRANS": label_encoders['MTRANS'].transform([MTRANS])[0],
}

input_df = pd.DataFrame([input_dict])

# Prediksi
if st.button("Prediksi"):
    input_data = input_df[X.columns]  # Samakan kolom
    result = model.predict(input_data)
    label = le_target.inverse_transform(result)[0]
    
    # Penjelasan kategori
    keterangan = {
        "Insufficient_Weight": "Berat badan kurang",
        "Normal_Weight": "Berat badan normal",
        "Overweight_Level_I": "Kelebihan berat badan tingkat I",
        "Overweight_Level_II": "Kelebihan berat badan tingkat II",
        "Obesity_Type_I": "Obesitas tingkat I (ringan)",
        "Obesity_Type_II": "Obesitas tingkat II (sedang)",
        "Obesity_Type_III": "Obesitas tingkat III (berat)"
    }

    penjelasan = keterangan.get(label, "Kategori tidak diketahui")
    st.success(f"Prediksi tingkat obesitas Anda: **{penjelasan}**  \n(Kode model: `{label}`)")


# Footer
st.caption("Dibuat dengan ❤️ oleh Streamlit dan Scikit-learn")
