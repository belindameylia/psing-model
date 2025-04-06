from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load ML model
model = pickle.load(open("../export_model/model_naivebayes.pkl", "rb"))

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict/")
def predict(data: InputData):
    prediction = model.predict([np.array(data.features)])
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)