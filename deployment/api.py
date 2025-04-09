from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn import preprocessing
from typing import List
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 untuk development, izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = pickle.load(open("../export_model/model_randomforest.pkl", "rb"))

# Load Y Value Encoder
produk_encoder = pickle.load(open("../export_model/produk.pkl", "rb"))

# Load X Value Encoder
umur_encoder = pickle.load(open("../export_model/umur.pkl", "rb"))
gender_encoder = pickle.load(open("../export_model/gender.pkl", "rb"))
domisili_encoder = pickle.load(open("../export_model/domisili.pkl", "rb"))
profesi_encoder = pickle.load(open("../export_model/profesi.pkl", "rb"))
status_perkawinan_encoder = pickle.load(open("../export_model/status_perkawinan.pkl", "rb"))
penghasilan_encoder = pickle.load(open("../export_model/penghasilan.pkl", "rb"))
persentase_tabungan_encoder = pickle.load(open("../export_model/persentase_tabungan.pkl", "rb"))

# Request Schema

class Tujuan(BaseModel):
    investasi: int
    simpanan_jangka_panjang: int
    kegiatan_sehari_hari: int
    tujuan_lainnya: int
  
class FormData(BaseModel):
    umur: str
    gender: str
    domisili: str
    profesi: str
    status_perkawinan: str
    tujuan: Tujuan
    penghasilan: str
    persentase_tabungan: str
    rate_fungsionalitas: int
    rate_admin: int
    rate_limit: int
    rate_bunga: int
    rate_setoran_awal: int
    rate_kebutuhan: int
  
# Response schemas
class Recommendation(BaseModel):
    class_name: str
    probability: str

class PredictionResult(BaseModel):
    predicted_class: str
    compatibility: str
    top_3_recommendations: List[Recommendation]

# --- SAFETY FUNCTION ---
def safe_label_transform(encoder, value: str):
    known_classes = encoder.classes_
    if value not in known_classes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid value '{value}'. Allowed values: {known_classes.tolist()}"
        )
    return encoder.transform([value])[0]

@app.post("/predict/")
def predict(data: FormData):
    try:
        # Preprocess Label Encode
        umur = np.array([[safe_label_transform(umur_encoder, data.umur)]])
        domisili = np.array([[safe_label_transform(domisili_encoder, data.domisili)]])
        gender = np.array([[safe_label_transform(gender_encoder, data.gender)]])
        status_perkawinan = np.array([[safe_label_transform(status_perkawinan_encoder, data.status_perkawinan)]])
        profesi = np.array([[safe_label_transform(profesi_encoder, data.profesi)]])
        penghasilan = np.array([[safe_label_transform(penghasilan_encoder, data.penghasilan)]])
        persentase_tabungan = np.array([[safe_label_transform(persentase_tabungan_encoder, data.persentase_tabungan)]])

        # Raw input
        investasi = np.array([[data.tujuan.investasi]])
        simpanan_jangka_panjang = np.array([[data.tujuan.simpanan_jangka_panjang]])
        kegiatan_sehari_hari = np.array([[data.tujuan.kegiatan_sehari_hari]])
        tujuan_lainnya = np.array([[data.tujuan.tujuan_lainnya]])
        
        # Tambahan 
        rate_fungsionalitas = np.array([[data.rate_fungsionalitas]])
        rate_admin = np.array([[data.rate_admin]])
        rate_limit = np.array([[data.rate_limit]])
        rate_bunga = np.array([[data.rate_bunga]])
        rate_setoran_awal = np.array([[data.rate_setoran_awal]])
        rate_kebutuhan = np.array([[data.rate_kebutuhan]])
        
        # Combine
        X = np.hstack([umur, domisili, gender, status_perkawinan, profesi, investasi, simpanan_jangka_panjang, kegiatan_sehari_hari, tujuan_lainnya, penghasilan, persentase_tabungan, rate_fungsionalitas, rate_admin, rate_limit, rate_bunga, rate_setoran_awal, rate_kebutuhan])
        
        # Predict
        predict_result = model.predict(X)[0]
        predict_label = produk_encoder.inverse_transform([predict_result])[0]
        
        predict_compability = model.predict_proba(X)[0]
        
        top_3_indices = predict_compability.argsort()[-3:][::-1]
        top_3_classes = produk_encoder.inverse_transform(top_3_indices)
        top_3_probs = predict_compability[top_3_indices]
        
        recommendations = [
            Recommendation(
                class_name=class_name,
                probability=f"{prob * 100:.2f}%"
            )
            for class_name, prob in zip(top_3_classes, top_3_probs)
        ]
        
        return PredictionResult(
            predicted_class=predict_label,
            compatibility=f"{np.max(predict_compability) * 100:.2f}%",
            top_3_recommendations=recommendations
        )
        
    except HTTPException as http_exc:
        raise http_exc  # Let FastAPI handle it properly
        
    except Exception as e:
        return{
            "error":str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)