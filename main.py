from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np 

# Load model
model = joblib.load('model/model.pkl')

app = FastAPI()

# Request body model
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction route
@app.post('/predict')
def predict(request: PredictRequest):
    data = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])
    prediction = model.predict(data)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return {'species': species_map[int(prediction[0])]}

# Root route
@app.get('/')
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}
