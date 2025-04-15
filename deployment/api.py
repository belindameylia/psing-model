from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn import preprocessing
from typing import List
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #change buat limit akses API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Load ML model RF, SVC, Extra Trees
# model = pickle.load(open("../export_model/model_svc.pkl", "rb"))

# Load LightGBM
model = lgb.Booster(model_file='../export_model/model_lightgbm.txt')

# # Load XGBoost
# model = xgb.XGBClassifier()
# model.load_model('../export_model/model_xgboost.json')

# # Load ML model ANN
# model = load_model("../export_model/model_ann.h5")

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
        
        # Combine
        X = np.hstack([umur, domisili, gender, status_perkawinan, profesi, investasi, simpanan_jangka_panjang, kegiatan_sehari_hari, tujuan_lainnya, penghasilan, persentase_tabungan])
        
        # Active jika ANN, LightGBM
        X = np.array(X, dtype=np.float32).reshape(1, -1)
        
        # # Predict Model lain
        # predict_result = model.predict(X)[0]
        # predict_label = produk_encoder.inverse_transform([predict_result])[0]
        # predict_compability = model.predict_proba(X)[0]
        
        # Predict Model ANN, LightGBM
        predict_proba = model.predict(X)  # (1, 5)
        predict_result = np.argmax(predict_proba, axis=1)[0]  # hasil index prediksi (misal 2)
        predict_label = produk_encoder.inverse_transform([predict_result])[0]
        predict_compability = predict_proba[0]
        
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