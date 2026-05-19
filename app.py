import streamlit as st
from groq import Groq
import numpy as np
import joblib
import os
import json


st.set_page_config(page_title="StressPredict API Backend", page_icon="🧠")

MODEL_PATH = "model_stress.pkl"
SCALER_PATH = "scaler.pkl"  

@st.cache_resource
def load_ml_components():
    m = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    s = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return m, s

model, scaler = load_ml_components()


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

NAMA_FITUR = [
    'Faktor Umur', 'Faktor Gender', 'Tahun Angkatan Kuliah', 'Durasi Waktu Belajar Harian', 
    'Beban Tekanan Ujian', 'Performa Akademik / IPK', 'Tingkat Kecemasan (Anxiety)', 
    'Kondisi Suasana Hati (Mood/Depresi)', 'Kualitas & Durasi Istirahat/Tidur', 
    'Rendahnya Aktivitas Fisik/Olahraga', 'Keterbatasan Dukungan Sosial', 
    'Durasi Paparan Layar Gadget (Screen Time)', 'Pola Konsumsi Internet Harian', 
    'Tekanan Finansial/Keuangan Mahasiswa', 'Tuntutan/Ekspektasi dari Keluarga', 
    'Tingkat Kejenuhan Akademik (Burnout)', 'Kondisi Kesehatan Mental Secara Umum'  
]

def ambil_rekomendasi_groq(status_stres, umur, jam_tidur, tekanan_ujian):
    try:
        u = int(umur) if str(umur).isdigit() else 21
        jt = int(jam_tidur) if str(jam_tidur).isdigit() else 7
        tu = int(tekanan_ujian) if str(tekanan_ujian).isdigit() else 5

        prompt = f"""
        Anda adalah seorang psikolog dan konselor akademik kampus yang profesional.
        Berikan 3 rekomendasi tindakan medis/psikologis yang konkret, singkat, dan menenangkan untuk mahasiswa dengan profil berikut:
        - Umur: {u} tahun
        - Hasil Prediksi Model ML: {str(status_stres)}
        - Jam Tidur: {jt} jam/malam
        - Skala Tekanan Ujian: {tu}/10

        Format output HARUS berupa JSON object dengan key "rekomendasi" yang berisi array string, contoh wajib:
        {{
            "rekomendasi": ["Rekomendasi 1", "Rekomendasi 2", "Rekomendasi 3"]
        }}
        Jangan berikan teks pembuka, penutup, atau tanda ```json . Langsung berikan format JSON murni saja.
        """

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            response_format={"type": "json_object"} 
        )
        
        raw_content = completion.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[1]
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("\n", 1)[0]
        raw_content = raw_content.strip("`").strip()

        data_json = json.loads(raw_content)
        return data_json.get("rekomendasi", [])
        
    except Exception as e:
        return [
            "Atur jadwal tidur malam minimal 7 jam untuk memulihkan energi otak.",
            "Sempatkan istirahat 5-10 menit setiap 50 menit belajar (Teknik Pomodoro).",
            "Diskusikan beban akademik dengan dosen pembimbing atau konselor kampus."
        ]

def proses_kalkulasi_stress(data):
    def safe_int(val, default=0): return int(val) if str(val).isdigit() else default
    def safe_float(val, default=0.0):
        try: return float(val)
        except: return default

    g_input = data.get('gender', 'Female')
    gender_encoded = 1 if g_input in ["Laki-laki", "Male"] else 0
    
    th_input = str(data.get('tahunAkademik', 'Tahun 1'))
    academic_year_encoded = 2 if '2' in th_input else (3 if '3' in th_input else (4 if '4' in th_input else 1))

    ipk_vue = safe_float(data.get('ipk'), 3.5)
    academic_performance_scaled = min(100.0, max(1.0, ipk_vue * 25.0))

    raw_features = np.array([[
        safe_int(data.get('umur'), 21), gender_encoded, academic_year_encoded,                        
        safe_float(data.get('jamBelajar'), 6.0), safe_int(data.get('tekananUjian'), 5),        
        academic_performance_scaled, safe_int(data.get('anxietyScore'), 5),        
        safe_int(data.get('depressionScore'), 5), safe_int(data.get('jamTidur'), 7),            
        safe_int(data.get('aktivitasFisik'), 3), safe_int(data.get('socialSupport'), 7),       
        safe_float(data.get('screenTime'), 4.0), safe_float(data.get('internetUsage'), 4.0),   
        safe_int(data.get('financialStress'), 4), safe_int(data.get('ekspektasiKeluarga'), 5),  
        safe_int(data.get('burnoutScore'), 5), safe_int(data.get('mentalHealthIndex'), 7)    
    ]], dtype=np.float32)

    status_terprediksi = "Stres Sedang (Moderate)"
    score_display = 7.0
    faktor_dominan = ["Beban Tekanan Ujian"]

    if model is not None and scaler is not None:
        try:
            scaled_features = scaler.transform(raw_features)
            prediction = model.predict(scaled_features)[0]
            categories = {0: "Stres Rendah (Low)", 1: "Stres Sedang (Moderate)", 2: "Stres Tinggi (High)"}
            status_terprediksi = categories.get(prediction, "Stres Sedang (Moderate)")
            
            if hasattr(model, "predict_proba"):
                score_display = float(round(model.predict_proba(scaled_features)[0][prediction] * 10, 1))
                
            importances = model.feature_importances_
            z_scores = scaled_features[0].copy()
            for idx in [8, 10, 16, 5]: z_scores[idx] = -z_scores[idx]

            kontribusi_fitur = np.maximum(0, z_scores) * importances
            indeks_teratas = np.argsort(kontribusi_fitur)[::-1][:2]
            faktor_dominan = [str(NAMA_FITUR[idx]) for idx in indeks_teratas if kontribusi_fitur[idx] > 0]
        except Exception as e:
            pass

    rekomendasi_ai = ambil_rekomendasi_groq(status_terprediksi, raw_features[0][0], raw_features[0][8], raw_features[0][4])

    return {
        "status": str(status_terprediksi),
        "score": score_display,
        "faktorDominan": faktor_dominan[:2],
        "rekomendasi": rekomendasi_ai
    }

query_params = st.query_params

if "payload" in query_params:
    try:
        raw_payload = query_params["payload"]
        data_input = json.loads(raw_payload)
        hasil_prediksi = proses_kalkulasi_stress(data_input)
        
        
        st.text(json.dumps(hasil_prediksi))
        st.stop()
    except Exception as e:
        st.text(json.dumps({"error": str(e)}))
        st.stop()

st.title("StressPredict - ML Deployment Server")
st.markdown("---")
if model is not None and scaler is not None:
    st.success("Server & Model XGBoost Berhasil Disinkronkan!")
else:
    st.error("Komponen Model (.pkl) gagal dimuat di server.")