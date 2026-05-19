from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import numpy as np
import joblib
import os
import json

app = FastAPI(title="Stress Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model_stress.pkl"
SCALER_PATH = "scaler.pkl"  

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": f"Berikan 3 rekomendasi psikologis singkat untuk mahasiswa {umur} thn, status stres: {status_stres}, tidur: {jam_tidur} jam, tekanan ujian: {tekanan_ujian}/10. Format JSON murni: {{\"rekomendasi\": [\"R1\", \"R2\", \"R3\"]}}"}],
            temperature=0.6,
            response_format={"type": "json_object"} 
        )
        data_json = json.loads(completion.choices[0].message.content.strip())
        return data_json.get("rekomendasi", [])
    except Exception as e:
        print(f"EROR GROQ: {e}")
        return [
            "Atur jadwal tidur malam minimal 7 jam untuk memulihkan energi otak.",
            "Sempatkan istirahat 5-10 menit setiap 50 menit belajar (Teknik Pomodoro).",
            "Diskusikan beban akademik dengan dosen pembimbing atau konselor kampus."
        ]

@app.post("/api/predict")
async def api_predict(request: Request):
    data = await request.json()
    
    def safe_int(val, default=0): return int(val) if str(val).isdigit() else default
    def safe_float(val, default=0.0):
        try: return float(val)
        except: return default

    gender_encoded = 1 if data.get('gender') in ["Laki-laki", "Male"] else 0
    th_input = str(data.get('tahunAkademik', 'Tahun 1'))
    academic_year_encoded = 2 if '2' in th_input else (3 if '3' in th_input else (4 if '4' in th_input else 1))
    academic_performance_scaled = min(100.0, max(1.0, safe_float(data.get('ipk'), 3.5) * 25.0))

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
        "status": status_terprediksi,
        "score": score_display,
        "faktorDominan": faktor_dominan[:2],
        "rekomendasi": rekomendasi_ai
    }

@app.get("/")
def health_check():
    return {"status": "FastAPI Backend is running perfectly!"}