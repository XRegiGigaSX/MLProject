import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info('Splitting train and test input data')

            # [:, :-1] signifies all rows and all columns except last column
            # [:, -1] signifies all rows and last column
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            # create dictionary of models to be used for training
            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBClassifier": XGBRegressor(), 
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,  models = models)
        
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model with accuracy above 60%.')

            logging.info('Best Model [{}] found on both train and test dataset.'.format(best_model_name))

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Best model saved as pickle file.")

            # Testing the best model on test dataset
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square 


        except Exception as e:
            raise CustomException(e, sys)
