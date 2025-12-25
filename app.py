
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


# data = Item(
#     longitude=122, 
#     latitude=96.5,
#     housing_median_age=45,
#     total_rooms=90,
#     total_bedrooms=90,
#     population=90,
#     households=90,
#     median_income=90,
#     ocean_proximity="inland"
#     )
