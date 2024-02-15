# Import necessary libraries
import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os
from utils import save_object,model_training

# Define a dataclass for Model Trainer configuration
@dataclass
class ModelTrainerConfig:
    model_path=os.path.join("artifacts","model.pkl")

# Define the ModelTrainer class
class ModelTrainer:
    def __init__(self):
        self.model_saving_path=ModelTrainerConfig()
        logging.info("Define the model saving path")
        # Initialize the config object

    def initiate_model_training(self,X_train_array,y_train_array,X_test_array,y_test_array):
        try:
            self.X_train_array=X_train_array
            self.y_train_array=y_train_array
            self.X_test_array=X_test_array
            self.y_test_array=y_test_array
            logging.info("Define the training and testing array for x and y")
            models={
                "RandomForestRegressor":RandomForestRegressor(),
                "ExtraTreeRegressor":ExtraTreeRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()}
            logging.info("Models Are defined Successfully.")

            param={
                
                "ExtraTreeRegressor":{},
                "RandomForestRegressor":{},
                "DecisionTreeRegressor":
                {'criterion':['squared_error','absolute_error'],
                'max_depth':np.arange(1,21).tolist()[0::2],
                'min_samples_split':np.arange(2,11).tolist()[0::2],
                'max_leaf_nodes':np.arange(3,26).tolist()[0::2]},
                "KNeighborsRegressor":{}
        #         "RandomForestClassifier":{
        #     'criterion':['log_loss', 'gini', 'entropy'],
        #             'max_features':['sqrt','log2'],
        #             'n_estimators': [32,64,128] },
        
        #         "DecisionTreeClassifier":{
        #    'criterion':['log_loss', 'gini', 'entropy'],
        #             'splitter':['best','random'],
        #             'max_features':['sqrt','log2']},
        
        #         "KNeighborsClassifier":{'n_neighbors' : [5,7,9,11,13,15],
        #        'weights' : ['uniform','distance'],
        #        'metric' : ['minkowski','euclidean','manhattan']},
        
        #         "LogisticRegression":{}
                                            }
            logging.info("Hyperparameter Are defined Successfully.")
            
            score_dict = model_training(param=param,models=models,X_train_array=X_train_array,y_train_array=y_train_array,X_test_array=X_test_array,y_test_array=y_test_array)
            best_model_name = max(list(score_dict))
            best_model_score = score_dict[best_model_name]
            best_model = models[best_model_name]
            logging.info(f"Best model has been found and best model is {best_model_name } with test accuracy of {best_model_score}")
            # print(f"This is the best model {best_model} and it has the score of { best_model_score}")
            save_object(best_model,self.model_saving_path.model_path)
            logging.info("Model has been save successfully")

        except Exception as e:
            raise CustomException(e)