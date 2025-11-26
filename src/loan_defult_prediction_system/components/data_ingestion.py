import os
import sys
from src.loan_defult_prediction_system.exception import CustomException
from src.loan_defult_prediction_system.logger import logging
import pandas as pd
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

class DataIngestion:
    def __init__(self, config_filepath=Path("config/config.yaml")):
        self.config_filepath = config_filepath
        self.config = self.read_yaml_config()
        self.ingestion_config = self.get_data_ingestion_config()

    def read_yaml_config(self) -> dict:
        try:
            with open(self.config_filepath) as yaml_file:
                content = yaml.safe_load(yaml_file)
                logging.info(f"yaml file: {self.config_filepath} loaded successfully")
                return content
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        
        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir']) 
        )

        return data_ingestion_config

    def download_file(self):
        try:
            dataset_name = self.ingestion_config.source_URL
            zip_download_dir = self.ingestion_config.root_dir
            
            os.makedirs(zip_download_dir, exist_ok=True)
            
            logging.info(f"Downloading data from {dataset_name} into {zip_download_dir}")
            
            # Using kaggle API to download dataset
            os.system(f"kaggle datasets download -d {dataset_name} -p {zip_download_dir} --unzip")
            
            logging.info(f"Downloaded data from {dataset_name} into {zip_download_dir}")

        except Exception as e:
            raise CustomException(e, sys)

    
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        obj.download_file()
    except Exception as e:
        logging.exception(e)
        raise e
