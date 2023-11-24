import os
import sys
from src.house_pricing.exception import CustomException
from src.house_pricing.logger import logging
from src.house_pricing.entity import DataIngestionConfig
import pandas as pd



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            train_set = pd.read_csv('data\house_train.csv')
            logging.info('Read the training dataset as dataframe')

            test_set = pd.read_csv('data\house_test.csv')
            logging.info('Read the testing dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and test data has been saved")
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    #data_transformation=DataTransformation()
    #train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



