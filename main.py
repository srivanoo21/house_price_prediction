from src.house_pricing.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.house_pricing.pipeline.stage_02_data_validation import DataValidationPipeline
from src.house_pricing.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.house_pricing.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from src.house_pricing.logger import logging


STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logging.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logging.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Model Training"
try:
    logging.info(f">>>> stage {STAGE_NAME} started <<<<")
    model_train = ModelTrainerPipeline()
    model_train.main()
    logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logging.exception(e)
    raise e