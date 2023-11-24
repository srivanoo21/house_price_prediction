from src.house_pricing.config.configuration import ConfigurationManager
from src.house_pricing.components.data_ingestion import DataIngestion
from src.house_pricing.logger import logging


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            #config = ConfigurationManager()
            #data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise e