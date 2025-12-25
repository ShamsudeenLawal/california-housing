import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

model = load_object("artifacts/model/model.joblib")
preprocessor = load_object("artifacts/preprocessor.joblib")

def predict(data):
    transformed_data = preprocessor.transform(data)
    print(transformed_data)
    print(model)
    pred = model.predict(transformed_data)
    
    return pred[0]
