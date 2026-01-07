
import os
import sys

# Add root directory to sys.path
sys.path.append(os.getcwd())

from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
from src.loan_defult_prediction_system.components.data_ingestion import DataIngestion
from src.loan_defult_prediction_system.components.data_transformation import DataTransformation
from src.loan_defult_prediction_system.components.model_trainer import ModelTrainer
from src.loan_defult_prediction_system.components.model_monitering import ModelMonitoring

if __name__ == "__main__":
    try:
        logging.info(">>>>> Test Training Pipeline Started (50 Samples) <<<<<")
        
        # 1. Data Ingestion
        logging.info("Step 1: Data Ingestion")
        obj = DataIngestion()
        # Using 50 samples to be safe for CV (5 samples is too unstable)
        train_data_path, test_data_path = obj.initiate_data_ingestion(sample_size=50)
        print(f"Data Ingestion Completed. Train path: {train_data_path}, Test path: {test_data_path}")

        # 2. Data Transformation
        logging.info("Step 2: Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        print("Data Transformation Completed.")

        # 3. Model Training
        logging.info("Step 3: Model Training")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model Training Completed. Best Model Accuracy: {accuracy*100:.2f}%")
        
        print(">>>>> Test Pipeline Completed Successfully <<<<<")

    except Exception as e:
        logging.error("Error in Test Training Pipeline")
        print(e)
