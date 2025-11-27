import os
import sys
from src.loan_defult_prediction_system.logger import logging
from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.components.data_ingestion import DataIngestion
from src.loan_defult_prediction_system.components.data_transformation import DataTransformation
from src.loan_defult_prediction_system.components.model_trainer import ModelTrainer
from src.loan_defult_prediction_system.components.model_monitering import ModelMonitoring

class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
            obj = DataIngestion()
            obj.download_file()
            logging.info("Exited the start_data_ingestion method of TrainingPipeline class")
            
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, train_path, test_path):
        try:
            logging.info("Entered the start_data_transformation method of TrainingPipeline class")
            obj = DataTransformation()
            train_arr, test_arr, _ = obj.initiate_data_transformation(train_path, test_path)
            logging.info("Exited the start_data_transformation method of TrainingPipeline class")
            return train_arr, test_arr
            
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            logging.info("Entered the start_model_training method of TrainingPipeline class")
            obj = ModelTrainer()
            obj.initiate_model_trainer(train_arr, test_arr)
            logging.info("Exited the start_model_training method of TrainingPipeline class")
            
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_monitoring(self):
        try:
            logging.info("Entered the start_model_monitoring method of TrainingPipeline class")
            obj = ModelMonitoring()
            obj.initiate_model_monitoring()
            logging.info("Exited the start_model_monitoring method of TrainingPipeline class")
            
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Started Training Pipeline")
            
            # Data Ingestion
            # Assuming download_file returns nothing but files are at known locations
            self.start_data_ingestion()
            
            # Define paths (should ideally come from config, but hardcoding for now based on known structure)
            train_path = os.path.join("artifacts", "data_ingestion", "train.csv")
            test_path = os.path.join("artifacts", "data_ingestion", "test.csv")
            
            # Data Transformation
            train_arr, test_arr = self.start_data_transformation(train_path, test_path)
            
            self.start_model_training(train_arr, test_arr)
            self.start_model_monitoring()
            
            logging.info("Completed Training Pipeline")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception("Exception occurred in TrainingPipeline")
        print(e)
