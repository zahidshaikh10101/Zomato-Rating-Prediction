from logger import logging
import os
import sys
from exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import mysql.connector as connection

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join(os.getcwd(),"artifacts","data.csv")
    train_data_path = os.path.join(os.getcwd(),"artifacts","train.csv")
    test_data_path = os.path.join(os.getcwd(),"artifacts","test.csv")
    logging.info("Gives the Raw data,Train and Test path to store the data.")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            mydb = connection.connect(host="localhost", database = 'zomato',user="root", passwd="Zarss@786",use_pure=True)
            logging.info("Setting the connection with the mysql for read the data")
            query = "select * from zomato.zomato_data;"
            logging.info(f"select the column which we need for the dataset: {query}")
            result_dataFrame = pd.read_sql(query,mydb)
            mydb.close()
            df = result_dataFrame
            logging.info("Read as Dataframes")
            logging.info(df.head())
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train Test split initiated")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info("Created the Artifacts folder for the save the train and test data")

            df.to_csv(self.ingestion_config.raw_data_path)
            train_data.to_csv(self.ingestion_config.train_data_path)
            test_data.to_csv(self.ingestion_config.test_data_path)
            
            logging.info("Data Ingestion is completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_ingestion_obj=DataIngestion()
    train_path,test_path=data_ingestion_obj.initiate_data_ingestion()