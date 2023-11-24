#from src.house_pricing.config.configuration import ConfigurationManager
from src.house_pricing.components.data_transformation import DataTransformation
from src.house_pricing.entity import DataTransformationConfig
from src.house_pricing.logger import logging


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            pass
            #config = ConfigurationManager()
            #data_validation_config = config.get_data_validation_config()
            data_transformation = DataTransformation()
            data_transformation.initiate_data_transformation()

        except Exception as e:
            raise e
