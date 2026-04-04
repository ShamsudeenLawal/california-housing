import os
import sys
import pandas as pd
import dataclasses
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging


@dataclasses.dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def run(self):
        logging.info("Starting data ingestion...")
        try:
            
            # specify data path: one can also consider passing this as an argument
            # data_path = os.path.join("notebooks", "data", "housing", "housing.csv")
            data_path = os.path.join("data", "housing", "housing.csv")
            
            # create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # load data
            housing = pd.read_csv(data_path)
            
            # save raw data
            housing.to_csv(self.ingestion_config.raw_data_path)

            # split data
            housing["income_category"] = pd.cut(housing["median_income"],
                                    bins=4, labels=range(4))
            
            housing_train, housing_test = train_test_split(housing, test_size=0.2,
                                               stratify=housing["income_category"],
                                               shuffle=True, random_state=42)
            
            # drop income category
            housing_train = housing_train.drop(columns=["income_category"])
            housing_test = housing_test.drop(columns=["income_category"])

            # save train and test data
            housing_train.to_csv(self.ingestion_config.train_data_path, index=False)
            housing_test.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data successfully ingested...")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as err:
            CustomException(err, sys) # type: ignore


