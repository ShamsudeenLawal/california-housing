
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
    tuner = ModelTuner()
    test_score = tuner.run(train_arr, test_arr)
    print("Test Score:", test_score)

if __name__ == "__main__":
    run()
