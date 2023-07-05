import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = (
        os.path.abspath(os.path.join(os.path.dirname("__file__"), "."))
        + "/data/train.csv"
    )
    test_data_path: str = (
        os.path.abspath(os.path.join(os.path.dirname("__file__"), "."))
        + "/data/test.csv"
    )
    raw_data_path: str = (
        os.path.abspath(os.path.join(os.path.dirname("__file__"), "."))
        + "/data/forest_data.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Read the dataset as dataframe")
            del df["Area"]
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True,
            )

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                train_set,
                test_set,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "main_":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_data, test_data))
