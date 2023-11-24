#from src.house_pricing.config.configuration import ConfigurationManager
from src.house_pricing.components.data_validation import DataValidation
from src.house_pricing.entity import DataValidationConfig
from src.house_pricing.logger import logging


class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            pass
            #config = ConfigurationManager()
            #data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=DataValidationConfig)
            data_validation.validate_all_files_exist()

        except Exception as e:
            raise e
