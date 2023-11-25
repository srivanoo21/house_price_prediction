from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.house_pricing.pipeline.prediction import PredictionPipeline


app = FastAPI()

@app.get("/", tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.post("/predict")
async def predict_route():
    try:
        obj = PredictionPipeline()
        obj.get_transformed_data()
        obj.predict_data()
        return Response("Prediction successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)