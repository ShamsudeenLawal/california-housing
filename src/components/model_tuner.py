import os
import sys
import sklearn
import numpy as np
from scipy.stats import randint
import dataclasses

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


models = {
    "linear_regression": sklearn.linear_model.LinearRegression(), # type: ignore
    "lasso": sklearn.linear_model.Lasso(), # type: ignore
    "ridge": sklearn.linear_model.Ridge(), # type: ignore
    "decision_tree": sklearn.tree.DecisionTreeRegressor(), # type: ignore
    "random_forest": sklearn.ensemble.RandomForestRegressor(), # type: ignore
    "knn": sklearn.neighbors.KNeighborsRegressor() # type: ignore
}

hyperparameters = {
                    "linear_regression": {
                            "fit_intercept": [True, False],
                            "positive": [True, False]
                            },

                    "lasso": {
                            "alpha": np.logspace(-4, 1, 20),
                            "fit_intercept": [True, False],
                            "max_iter": [1000, 5000],
                            "selection": ["cyclic", "random"],
                            "positive": [True, False],
                            },

                    "ridge": {
                        "alpha": np.logspace(-4, 1, 20),
                        "fit_intercept": [True, False],
                        "max_iter": [1000, 5000],
                        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
                        "positive": [True, False]
                        },

                    "decision_tree": {
                        "max_depth": [None, 5, 10, 20, 40],
                        "min_samples_split": [2, 5, 10, 20],
                        "min_samples_leaf": [1, 5, 10],
                        "max_features": [None, "sqrt", "log2"],
                        },

                    "random_forest": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        "max_depth": [None, 10, 20, 30, 40, 50],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": [None, "sqrt", "log2"]
                        },
                    "knn": {
                        "n_neighbors": randint(3, 51),
                        "weights": ["uniform", "distance"],
                        "metric": ["euclidean", "manhattan", "minkowski"],
                        "p": [1, 2],
                        "leaf_size": randint(20, 60),
                    }
                }


scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}


def cross_validate_models(train_array: np.array, model_dict: dict, cv = None, metric="rmse"):
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn import metrics

    model_scores = {}
    # cross-validate each model
    try:
        logging.info("Starting model cross-validation...")
        for i, (model_name, model_obj) in enumerate(model_dict.items()):
            predictions = cross_val_predict(estimator=model_obj, X=train_array[:, :-1], y=train_array[:, -1], cv=cv)
            mae = metrics.mean_absolute_error(train_array[:, -1], predictions)
            rmse = metrics.root_mean_squared_error(train_array[:, -1], predictions)
            r2 = metrics.r2_score(train_array[:, -1], predictions)
            logging.info(f"Model: {model_name} | MAE: {mae} | RMSE: {rmse} | R2: {r2}")

            model_scores[model_name] = {"rmse": rmse, "mae": mae, "r2": r2}

        # get the best performing model
        sorted_model_scores = sorted(
            model_scores.items(),
            key=lambda item: item[1][metric.lower()],
            reverse=False if metric.lower() in ["rmse", "mae"] else True
        )

        logging.info(f"Best model after cross-validation: {sorted_model_scores[0][0]} with score: {sorted_model_scores[0][1]}")
        return sorted_model_scores[0][0]
    
    except Exception as err:
        raise CustomException(err, sys) # type: ignore


@dataclasses.dataclass
class ModelTunerConfig:
    model_path: str = os.path.join("artifacts", "model", "model.joblib")


class ModelTuner:
    def __init__(self):
        self.trainer_config = ModelTunerConfig()

    def run(self, train_array, test_array, metric="rmse", cv=None):
        logging.info("Starting Model Training...")
        try:
            # cross-validation
            best_model: str = cross_validate_models(train_array, model_dict=models, cv=cv, metric=metric)
            # tune best performing model
            from sklearn.model_selection import RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=models[best_model],
                               param_distributions=hyperparameters[best_model],
                               n_iter=3, cv=cv, refit=True, scoring=scoring[metric],
                               random_state=42,)
            
            random_search.fit(train_array[:, :-1], train_array[:, -1])
            logging.info(f"Best hyperparameters for {best_model}: {random_search.best_params_}")
            # evaluate the model
            test_score = np.abs(random_search.score(test_array[:, :-1], test_array[:, -1]))
            logging.info(f"Test {metric} for the best model {best_model}: {test_score}")
            
            # saving model
            from src.utils import save_object
            save_object(file_path=self.trainer_config.model_path,
                        obj=random_search.best_estimator_)
            return test_score
        
        except Exception as err:
            raise CustomException(err, sys) # type: ignore
