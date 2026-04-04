
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from src.pipelines.predict_pipeline import predict

class Item(BaseModel):
    longitude: Union[float, int]
    latitude: Union[float, int]
    housing_median_age: Union[float, int]
    total_rooms: Union[float, int]
    total_bedrooms: Union[float, int]
    population: Union[float, int]
    households: Union[float, int]
    median_income: Union[float, int]
    ocean_proximity: str

app = FastAPI()

@app.get("/")
def home():
    return {"Health Check": "Ok"}

@app.post("/predict")
def get_prediction(data: Item):
    columns = list(data.model_dump().keys())
    x_list = list(data.model_dump().values())
    x_df = pd.DataFrame([x_list], columns=columns)
    pred = predict(x_df)

    return {"prediction": pred}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", port=8080, reload=True)
