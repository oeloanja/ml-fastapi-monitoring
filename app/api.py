from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
from .schemas import loan, Rating, feature_names


ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
model = load(ROOT_DIR/ 'artifacts/model.joblib')

@app.get('/')

def root():
    return "loan_status Ratings"

@app.post("/predict" , response_model = Rating)
def predict(response: Response, sample : loan):
    sample_dict = sample.dict()
    features = np.arrary([sample_dict[f] for f in feature_names]).reshape(-1, 1)
    prediction = model.predict(features)[0]
    response.headers['X-model-scale'] = str([prediction])
    return Rating(loan_status = prediction)

@app.get('/healthcheck')
def healthcheck()
    return{'status' : 'ok'}

