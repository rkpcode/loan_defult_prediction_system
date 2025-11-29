import os
import sys
from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.loan_defult_prediction_system.utils import read_train_data

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str = None, sample_size: int = None):
        """
        Initiate data ingestion process
        Args:
            file_path: Deprecated, kept for backward compatibility
            sample_size: If provided, sample N rows for testing (default: None = use full dataset)
        """
        logging.info("Starting data ingestion process")
        try:
            # Read the dataset (full or sampled)
            df = read_train_data(sample_size=sample_size)
            logging.info(f"Dataset read successfully: {len(df)} rows")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Data split into training and testing sets")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)