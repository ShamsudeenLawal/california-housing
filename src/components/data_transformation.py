import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn import compose
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging


# cluster similarity transformation
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X=X, Y=self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i}" for i in range(self.n_clusters)]


# ratio transformation pipeline
def column_ratio(X):
  return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
  return ["ratio"] # feature names out


def ratio_pipeline():
  return pipeline.make_pipeline(
    SimpleImputer(strategy="median"),
    preprocessing.FunctionTransformer(func=column_ratio, feature_names_out=ratio_name),
    preprocessing.StandardScaler()
  )


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.joblib')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def run(self, train_path: str, test_path: str):
        try:
            logging.info("Data Transformation initiated")
            # Add data transformation logic here
            # read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # separate variables, features and target types
            # the target variable is included in the NUM_VARS as it is a numerical variable
            NUM_VARS = train_df.select_dtypes(include=["float64"]).columns.tolist()
            CAT_VARS = train_df.select_dtypes(include=["object"]).columns.tolist()

            LABEL = ["median_house_value"]
            NUM_FEATURES = [NUM_FEAT for NUM_FEAT in NUM_VARS if NUM_FEAT != LABEL[0]]
            CAT_FEATURES = [CAT_FEAT for CAT_FEAT in CAT_VARS if CAT_FEAT != LABEL[0]]
            train_features = train_df[NUM_FEATURES + CAT_VARS]
            train_labels = train_df[LABEL]

            test_features = test_df[NUM_FEATURES + CAT_VARS]
            test_labels = test_df[LABEL]

            # log transformation pipeline
            log_pipeline = pipeline.make_pipeline(
            SimpleImputer(strategy="median"),
            preprocessing.FunctionTransformer(np.log, feature_names_out="one-to-one"),
            preprocessing.StandardScaler()
            )

            # cluster similarity pipeline
            cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=43)

            # default pipeline (simple imputer and scaler features)
            default_num_pipeline = pipeline.make_pipeline(
            SimpleImputer(strategy="median"),
            preprocessing.StandardScaler()
            )

            # cat pipeline
            cat_pipeline = pipeline.make_pipeline(SimpleImputer(strategy="most_frequent"),
                                        preprocessing.OneHotEncoder(handle_unknown="ignore"))


            # combine all transformation pipelines
            preprocessor = compose.ColumnTransformer([
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                    "households", "median_income"]),
            ("geo", cluster_simil, ["latitude", "longitude"]),
            ("cat", cat_pipeline, compose.make_column_selector(dtype_include=object))
            ],
            remainder=default_num_pipeline) # one column remaining: housing_median_age

            # apply preprocessing transformer
            Xtrain = preprocessor.fit_transform(train_features)
            Xtest = preprocessor.transform(test_features)

            # save the preprocessor object
            import joblib
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_obj_file_path)

            # create final train and test arrays
            train_labels = train_labels.values
            test_labels = test_labels.values
            train_arr = np.c_[Xtrain, train_labels]
            test_arr = np.c_[Xtest, test_labels]

            logging.info("Data Transformation completed")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as err:
            raise CustomException(err, sys) # type: ignore
