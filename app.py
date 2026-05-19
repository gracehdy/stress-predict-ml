import streamlit as st
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
import uvicorn
import threading
import json

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
    'Faktor Umur', 
    'Faktor Gender', 
    'Tahun Angkatan Kuliah', 
    'Durasi Waktu Belajar Harian', 
    'Beban Tekanan Ujian', 
    'Performa Akademik / IPK', 
    'Tingkat Kecemasan (Anxiety)', 
    'Kondisi Suasana Hati (Mood/Depresi)', 
    'Kualitas & Durasi Istirahat/Tidur', 
    'Rendahnya Aktivitas Fisik/Olahraga', 
    'Keterbatasan Dukungan Sosial', 
    'Durasi Paparan Layar Gadget (Screen Time)', 
    'Pola Konsumsi Internet Harian', 
    'Tekanan Finansial/Keuangan Mahasiswa', 
    'Tuntutan/Ekspektasi dari Keluarga', 
    'Tingkat Kejenuhan Akademik (Burnout)', 
    'Kondisi Kesehatan Mental Secara Umum'  
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
        
        raw_content = completion.choices[0].message.content
        
        cleaned_content = raw_content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.split("\n", 1)[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit("\n", 1)[0]
        cleaned_content = cleaned_content.strip("`").strip()

        data_json = json.loads(cleaned_content)
        return data_json.get("rekomendasi", [])
        
    except Exception as e:
        print(f"EROR GROQ NYATA: {e}")
        return [
            "Atur jadwal tidur malam minimal 7 jam untuk memulihkan energi otak.",
            "Sempatkan istirahat 5-10 menit setiap 50 menit belajar (Teknik Pomodoro).",
            "Diskusikan beban akademik dengan dosen pembimbing atau konselor kampus."
        ]

def proses_kalkulasi_stress(data):
    def safe_int(val, default=0):
        return int(val) if str(val).isdigit() else default

    def safe_float(val, default=0.0):
        try: return float(val)
        except: return default

    g_input = data.get('gender', 'Female')
    gender_encoded = 1 if g_input in ["Laki-laki", "Male"] else 0
    
    th_input = str(data.get('tahunAkademik', 'Tahun 1'))
    if '1' in th_input: academic_year_encoded = 1
    elif '2' in th_input: academic_year_encoded = 2
    elif '3' in th_input: academic_year_encoded = 3
    elif '4' in th_input: academic_year_encoded = 4
    else: academic_year_encoded = 1

    ipk_vue = safe_float(data.get('ipk'), 3.5)
    academic_performance_scaled = min(100.0, max(1.0, ipk_vue * 25.0))

    raw_features = np.array([[
        safe_int(data.get('umur'), 21),               
        gender_encoded,                               
        academic_year_encoded,                        
        safe_float(data.get('jamBelajar'), 6.0),      
        safe_int(data.get('tekananUjian'), 5),        
        academic_performance_scaled,                  
        safe_int(data.get('anxietyScore'), 5),        
        safe_int(data.get('depressionScore'), 5),     
        safe_int(data.get('jamTidur'), 7),            
        safe_int(data.get('aktivitasFisik'), 3),      
        safe_int(data.get('socialSupport'), 7),       
        safe_float(data.get('screenTime'), 4.0),      
        safe_float(data.get('internetUsage'), 4.0),   
        safe_int(data.get('financialStress'), 4),     
        safe_int(data.get('ekspektasiKeluarga'), 5),  
        safe_int(data.get('burnoutScore'), 5),        
        safe_int(data.get('mentalHealthIndex'), 7)    
    ]], dtype=np.float32)

    faktor_dominan = []

    
    if model is not None and scaler is not None:
        try:
            scaled_features = scaler.transform(raw_features)
            
            prediction = model.predict(scaled_features)[0]
            categories = {0: "Stres Rendah (Low)", 1: "Stres Sedang (Moderate)", 2: "Stres Tinggi (High)"}
            status_terprediksi = categories.get(prediction, "Stres Sedang (Moderate)")
            
            if hasattr(model, "predict_proba"):
                prob_persen = model.predict_proba(scaled_features)[0][prediction]
                score_display = float(round(prob_persen * 10, 1))
            else:
                score_display = 7.5
                
            importances = model.feature_importances_
            z_scores = scaled_features[0].copy()
        
            indeks_terbalik = [8, 10, 16]
            for idx in indeks_terbalik:
                z_scores[idx] = -z_scores[idx]
                
            z_scores[5] = -z_scores[5]

            fitur_memburuk = np.maximum(0, z_scores)
            kontribusi_fitur = fitur_memburuk * importances
            indeks_teratas = np.argsort(kontribusi_fitur)[::-1][:2]
            
            faktor_dominan = [str(NAMA_FITUR[idx]) for idx in indeks_teratas if kontribusi_fitur[idx] > 0]

        except Exception as e:
            print(f"Error dalam Pipeline ML: {e}")
            status_terprediksi = "Stres Sedang (Moderate)"
            score_display = 5.5

    if not faktor_dominan:
        if safe_int(data.get('tekananUjian'), 5) > 6:
            faktor_dominan = ["Beban Tekanan Ujian", "Tingkat Kejenuhan Akademik (Burnout)"]
        else:
            faktor_dominan = ["Durasi Waktu Belajar Harian", "Pola Aktivitas Akademik"]

    rekomendasi_ai = ambil_rekomendasi_groq(
        status_stres=status_terprediksi,
        umur=raw_features[0][0],
        jam_tidur=raw_features[0][8],
        tekanan_ujian=raw_features[0][4]
    )

    return {
        "status": str(status_terprediksi),
        "score": score_display,
        "faktorDominan": faktor_dominan[:2],
        "rekomendasi": rekomendasi_ai
    }

st.title("StressPredict - Automated XGBoost Server")
st.write("Mendukung pipeline Standard Scaling otomatis untuk data Vue 3.")

if model is not None and scaler is not None:
    st.success("Model XGBoost & Scaler berhasil disinkronkan ke memori cloud!")
elif model is not None:
    st.warning("Model terdeteksi, tapi file 'scaler.pkl' hilang dari folder!")
else:
    st.error("Komponen ML tidak lengkap di folder.")

if "api_server_active" not in st.session_state:
    api_app = FastAPI()
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api_app.post("/api/predict")
    async def api_predict(request: Request):
        payload = await request.json()
        return proses_kalkulasi_stress(payload)

    threading.Thread(target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=8000), daemon=True).start()
    st.session_state["api_server_active"] = True