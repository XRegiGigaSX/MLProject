import os
import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# dataclass allows us to directly declare variables without using __init__ function
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component.")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Data successfully read as dataframe.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path))

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test data splitting initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion is completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
