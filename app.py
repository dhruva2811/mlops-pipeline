
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Request schema
class InputFeatures(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "MLOps model is running!"}

@app.post("/predict")
def predict(data: InputFeatures):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr).tolist()
    return {"prediction": prediction}
"# Triggering CI" 
