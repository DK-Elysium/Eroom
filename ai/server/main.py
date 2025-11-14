from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    x: float

@app.post("/predict")
def predict(req: PredictRequest):
    result = req.x * 2   # 단순 테스트용
    return {"result": result}

@app.get("/")
def root():
    return {"message": "FastAPI server is running!"}