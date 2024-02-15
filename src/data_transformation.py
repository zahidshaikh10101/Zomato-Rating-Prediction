# Import necessary libraries
from logger import logging
from exception import CustomException
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer, OrdinalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from utils import save_object
import numpy as np

# Define a dataclass to store the preprocessor path
@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")
    # Define the preprocessor path
    logging.info("Define the preprocessor path")

# Define the custom label binarizer class
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = OrdinalEncoder(*args, **kwargs)
        # Initialize the encoder

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.classes_ = self.encoder.categories_[0]
        # Fit the encoder to the data
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)
        # Transform the data using the encoder

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = DataTransformationConfig()
        # Initialize the preprocessor path

    def get_preprocessor(self):
        try:
            num_features = ["votes", "cost_for_two"]
            cat_features = ["online", "reservations", "rest_type", "type", "location"]
            logging.info("Define the numerical and categorical features")

            num_pipeline = Pipeline(steps=[
                ("imputing", SimpleImputer(strategy="mean")),
                ("scaling", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputing", SimpleImputer(strategy="most_frequent")),
                ("encoding", MyLabelBinarizer()),
                # ("scaler", StandardScaler(with_mean=False))  # Optional scaler
            ])

            logging.info("Define the numeric and categorical pipelines")
            logging.info(f'Categorical columns: {cat_features}')
            logging.info(f'Numerical columns: {num_features}')

            preprocessor = ColumnTransformer([
                ("num_features", num_pipeline, num_features),
                ("cat_features", cat_pipeline, cat_features)
            ])

            logging.info("Define the preprocessor")
            # No need to print the preprocessor itself
            return preprocessor

        except Exception as e:
            raise CustomException(e, "Error occurred in get_preprocessor")

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            Target = "rating"
            y = train_data[Target]
            X = train_data.drop(Target, axis=1)

            preprocessor = self.get_preprocessor()

            # Split data after fitting the preprocessor on training data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

            preprocessor.fit(X_train)  # Fit only on training data
            logging.info("Fetch the preprocessor Successfully.")
            save_object(preprocessor,self.preprocessor_path.preprocessor_path)
            logging.info("Save the object Successfully")

            # Transform the data using the preprocessor and convert to a numpy array
            X_train_array = preprocessor.transform(X_train)
            y_train_array = np.array(y_train)
            X_test_array = preprocessor.transform(X_test)
            y_test_array = np.array(y_test)
            logging.info(f"X_train Shape: {X_train_array.shape}")
            logging.info(f"X_test Shape: {X_test_array.shape}")

            return (X_train_array,y_train_array,X_test_array,y_test_array)
        except Exception as e:
            raise CustomException(e)

if __name__=="__main__":
       data_transformation_obj=DataTransformation()
       data_transformation_obj.get_preprocessor()
       data_transformation_obj.initiate_data_transformation(r"artifacts\train.csv",
                                                           r"artifacts\test.csv")
