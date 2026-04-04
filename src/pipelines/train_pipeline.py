
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_tuner import ModelTuner

def run():

    # ingest the data
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.run() # type: ignore
    # transform the data
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor = transformer.run(train_data_path, test_data_path) # type: ignore
    # train/tune model
    cv = 3
    n_iter=20
    metric_to_optimizer = "r2"
    tuner = ModelTuner()
    metric_dicts = tuner.run(train_arr, test_arr, cv=cv, n_iter=n_iter)
    print("Best model metrics:", metric_dicts)

if __name__ == "__main__":
    run()
