from logger import logging
from exception import CustomException
import numpy as np
import pandas as pd
from utils import load_object

class Prediction:

    def __init__(self,online,reservations,votes,location,rest_type,cost_for_two,type):
        self.online=online
        self.reservations=reservations
        self.votes=votes
        self.location=location
        self.rest_type=rest_type
        self.cost_for_two=cost_for_two
        self.type=type
        logging.info("Getting all the data")

    def get_dataframe(self):
        try:

            data_points=[self.online,
                        self.reservations,
                        self.votes,
                        self.location,
                        self.rest_type,
                        self.cost_for_two,
                        self.type]
            dp=np.array(data_points).reshape(1,-1)
            logging.info(f"Get the data points and reshape them for fit the dataframe and data points are: {dp}")
            column_name=["online","reservations","votes","location","rest_type","cost_for_two","type"]
            input_df=pd.DataFrame(dp,columns=column_name)
            logging.info(f"Create the dataframe for the input of Model ")
            return input_df
        except Exception as e:
            raise CustomException(e)
        
    def Predict_rating(self,preprocessor_path,model_path):
        try:
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            new_df=self.get_dataframe()
            new_df_array=preprocessor.transform(new_df)
            predict=model.predict(new_df_array)
            logging.info(f"Predicted Rating: {predict[0]}")
            return predict[0]
        except Exception as e:
            raise CustomException(e)
        


prediction_obj=Prediction("Yes","Yes",775,"Banashankari","Casual Dining" ,1800,"Buffet")
preprocessor_path=r"D:\projects\Zomato_Rating_Prediction\artifacts\Preprocessor.pkl"
model_path=r"D:\projects\Zomato_Rating_Prediction\artifacts\model.pkl"
prediction_obj.Predict_rating(preprocessor_path=preprocessor_path,model_path=model_path)