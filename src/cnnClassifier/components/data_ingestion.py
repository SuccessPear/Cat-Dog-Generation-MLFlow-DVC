import os
import shutil
import zipfile
import kagglehub, gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def download_from_kaggle(self):
        """
        Download, extract and move to appropriate folder
        """
        # Download latest version
        dataset_url = self.config.source_URL
        unzip_path = self.config.unzip_dir

        path = kagglehub.dataset_download(dataset_url)
        logger.info(f"Downloading data from {dataset_url} into file {path}")

        if not os.path.exists(os.path.join(unzip_path, "PetImages")):
            os.makedirs(unzip_path, exist_ok=True)
            shutil.copytree(os.path.join(path, "PetImages"), os.path.join(unzip_path, "PetImages"))
            logger.info(f"Move dataset from cache {path} to {unzip_path}")
        else:
            logger.info(f"{os.path.join(unzip_path, "PetImages")} already exist")
