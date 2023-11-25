from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "data_ingestion", "house_training_data.csv")
    test_data_path: str = os.path.join('artifacts', "data_ingestion", "house_testing_data.csv")


@dataclass
class DataValidationConfig:
    ALL_REQUIRED_FILES: list
    root_dir: str = os.path.join("artifacts", "data_validation")
    STATUS_FILE: str = "status.txt"


@dataclass
class DataTransformationConfig:
    train_data_path = os.path.join('artifacts', "data_ingestion", "house_training_data.csv")
    test_data_path = os.path.join('artifacts', "data_ingestion", "house_testing_data.csv")
    preprocessed_train_data_path = os.path.join('artifacts', "data_transformation", "preprocesed_train_data.csv")
    preprocessed_test_data_path = os.path.join('artifacts', "data_transformation", "preprocesed_test_data.csv")


@dataclass
class ModelTrainerConfig:
    target_column: str = "SalePrice"
    preprocessed_train_data_path = os.path.join('artifacts', "data_transformation", "preprocesed_train_data.csv")
    trained_model_file_path: str = os.path.join("artifacts", "model_train", "model.pkl")


@dataclass
class PredictionConfig:
    predicted_column: str = "SalePrice"
    preprocessed_test_data_path = os.path.join('artifacts', "data_transformation", "preprocesed_test_data.csv")
    trained_model_file_path: str = os.path.join("artifacts", "model_train", "model.pkl")
    predicted_data_path = os.path.join("artifacts", "data_prediction", "predicted_data.csv")