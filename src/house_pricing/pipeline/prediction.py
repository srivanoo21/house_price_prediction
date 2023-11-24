#from src.house_pricing.config.configuration import ConfigurationManager
from src.house_pricing.components.model_trainer import ModelTrainer
from src.house_pricing.entity import PredictionConfig
from src.house_pricing.components.data_transformation_prediction import DataTransformation
from src.house_pricing.logger import logging
from src.house_pricing.utils import save_object, load_object


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionConfig()


    def get_transformed_data(self):
        '''
        get the transformed data for the prediction'''

        try:
            #config = ConfigurationManager()
            #data_validation_config = config.get_data_validation_config()
            data_transformation = DataTransformation()
            data_transformation.initiate_data_transformation()

        except Exception as e:
            raise e
        

    def predict_data(self):
        '''
        perform prediction on test data and return the output along with the predicted values'''

        try:
            test_data = self.config.preprocessed_test_data_path
            logging.info("Test data to be predicted is fetched")
                         
            model_path = self.config.trained_model_file_path
            model = load_object(file_path = model_path)
            logging.info("Model is loaded from the artifacts")

            predicted_column = self.config.predicted_column
            
            predicted_data = model.predict(test_data)
            logging.info("Prediction is performed on the test data")

            test_data[predicted_column] = predicted_data
            logging.info("Predicted data is now added in the test dataset")

            save_object(
                file_path = self.config.predicted_data_path,
                obj = test_data
            )
            logging.info("Test dataset after the prediction is now saved successfully in the artifacts")

        except Exception as e:
            raise e
