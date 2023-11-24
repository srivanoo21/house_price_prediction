#from src.house_pricing.config.configuration import ConfigurationManager
from src.house_pricing.components.model_trainer import ModelTrainer
from src.house_pricing.entity import ModelTrainerConfig
from src.house_pricing.logger import logging


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            pass
            #config = ConfigurationManager()
            #data_validation_config = config.get_data_validation_config()
            model_train = ModelTrainer()
            model_train.initiate_model_trainer()

        except Exception as e:
            raise e
