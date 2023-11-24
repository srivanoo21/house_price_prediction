import os
from src.house_pricing.logger import logging
from src.house_pricing.entity import DataValidationConfig


class DataValidation:
    
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion"))
            self.config.ALL_REQUIRED_FILES = ["house_training_data.csv", "house_testing_data.csv"]

            # create the status file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logging.info("Directory created for data validation")

            status_file = os.path.join(self.config.root_dir, self.config.STATUS_FILE)

            for file in all_files:
                if (file not in self.config.ALL_REQUIRED_FILES):
                    validation_status = False
                    with open(status_file, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                    return validation_status
                else:
                    validation_status = True
                    with open(status_file, "w") as f:
                        f.write(f"Validation status: {validation_status}")

            logging.info("Validation status has been updated")
            
        except Exception as e:
            raise e
        
        return validation_status
        