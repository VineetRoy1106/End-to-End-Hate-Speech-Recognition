import os
import sys
from zipfile import ZipFile
from hate.logger.logger import logger
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config

    def unzip_and_prepare(self):
        try:
            logger.info("Starting unzip_and_prepare method.")

            # Check ZIP exists
            if not os.path.exists(self.config.ZIP_FILE_PATH):
                raise FileNotFoundError(f"ZIP file not found: {self.config.ZIP_FILE_PATH}")

            with ZipFile(self.config.ZIP_FILE_PATH, "r") as zip_ref:
                # Extract each file and map to target paths
                for f in zip_ref.namelist():
                    if "raw" in f.lower():
                        target = self.config.NEW_DATA_ARTIFACTS_FILE
                    elif "imbalanced" in f.lower():
                        target = self.config.DATA_ARTIFACTS_FILE
                    else:
                        continue

                    # Delete target if exists
                    if os.path.exists(target):
                        os.remove(target)

                    # Extract to temporary location first, then rename
                    zip_ref.extract(f, self.config.DATA_INGESTION_DIR)
                    temp_path = os.path.join(self.config.DATA_INGESTION_DIR, f)
                    os.rename(temp_path, target)

            logger.info("ZIP extracted and files prepared successfully.")
            return self.config.DATA_ARTIFACTS_FILE, self.config.NEW_DATA_ARTIFACTS_FILE

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            imbalanced_file, raw_file = self.unzip_and_prepare()
            artifacts = DataIngestionArtifacts(
                imbalanced_data_file_path=imbalanced_file,
                raw_data_file_path=raw_file
            )
            logger.info(f"DataIngestionArtifacts created: {artifacts}")
            return artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
