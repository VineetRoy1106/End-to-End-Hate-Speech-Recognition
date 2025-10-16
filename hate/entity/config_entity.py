# # from dataclasses import dataclass

# # @dataclass
# # class DataIngestionConfig:
# #     ZIP_FILE_PATH: str = r"C:\Users\NLP-PIPELINE\dataset.zip"   # ✅ actual dataset location
# #     ZIP_FILE_DIR: str = r"C:\Users\NLP-PIPELINE\artifacts\unzipped"
# #     DATA_ARTIFACTS_DIR: str = r"C:\Users\NLP-PIPELINE\artifacts\data_ingestion\imbalanced"
# #     NEW_DATA_ARTIFACTS_DIR: str = r"C:\Users\NLP-PIPELINE\artifacts\data_ingestion\raw"


# # from dataclasses import dataclass

# # @dataclass
# # class DataIngestionConfig:
# #     ZIP_FILE_PATH: str = r"C:\Users\NLP-PIPELINE\dataset.zip"   # ✅ actual dataset location
# #     ZIP_FILE_DIR: str = 
# #     DATA_ARTIFACTS_DIR: str = r"C:\Users\NLP-PIPELINE\artifacts\data_ingestion\imbalanced"
# #     NEW_DATA_ARTIFACTS_DIR: str = r"C:\Users\NLP-PIPELINE\artifacts\data_ingestion\raw"


# import os
# from hate.constants import (
#     ROOT_DIR_KEY,
#     ARTIFACTS_DIR,
#     ZIP_FILE_NAME,
#     DATA_INGESTION_ARTIFACTS,
#     DATA_INGESTION_IMBALANCE_DATA_FILE,
#     DATA_INGESTION_RAW_DATA_FILE
# )

# class DataIngestionConfig:
#     def __init__(self):
#         # Optional: if using a cloud bucket
#         # self.BUCKET_NAME = None  # or set your bucket name
#         self.ZIP_FILE_NAME = ZIP_FILE_NAME

#         # Base artifact directory for this run
#         self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
#             ROOT_DIR_KEY, ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS
#         )

#         # Directory for imbalanced data
#         self.DATA_ARTIFACTS_DIR: str = os.path.join(
#             self.DATA_INGESTION_ARTIFACTS_DIR, "imbalanced"
#         )

#         # Directory for raw data
#         self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(
#             self.DATA_INGESTION_ARTIFACTS_DIR, "raw"
#         )

#         # Directory where ZIP will be extracted
#         # self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, "unzipped")

#         # Full path to ZIP file
#         self.ZIP_FILE_PATH = os.path.join(ROOT_DIR_KEY, self.ZIP_FILE_NAME)

#         # ✅ Ensure directories exist
#         os.makedirs(self.DATA_ARTIFACTS_DIR, exist_ok=True)
#         os.makedirs(self.NEW_DATA_ARTIFACTS_DIR, exist_ok=True)
#         # os.makedirs(self.ZIP_FILE_DIR, exist_ok=True)


import os
from hate.constants import (
    ROOT_DIR_KEY,
    ARTIFACTS_DIR,
    ZIP_FILE_NAME,
    DATA_INGESTION_ARTIFACTS
)
from dataclasses import dataclass
from hate.constants import *
import os

# @dataclass
# class DataIngestionConfig:
#     def __init__(self):
#         # ZIP file name
#         self.ZIP_FILE_NAME = ZIP_FILE_NAME

#         # Base artifact directory for this run
#         self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(
#             ROOT_DIR_KEY, ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS
#         )

#         # Directory for imbalanced data
#         self.DATA_ARTIFACTS_DIR: str = os.path.join(
#             self.DATA_INGESTION_ARTIFACTS_DIR, "imbalanced"
#         )

#         # Directory for raw data
#         self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(
#             self.DATA_INGESTION_ARTIFACTS_DIR, "raw"
#         )

#         # Full path to ZIP file
#         self.ZIP_FILE_PATH = os.path.join(ROOT_DIR_KEY, self.ZIP_FILE_NAME)

#         # ✅ Ensure directories exist
#         os.makedirs(self.DATA_ARTIFACTS_DIR, exist_ok=True)
#         os.makedirs(self.NEW_DATA_ARTIFACTS_DIR, exist_ok=True)


import os
from datetime import datetime
from dataclasses import dataclass


# Timestamp for artifact versioning
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Root project directory
ROOT_DIR = os.getcwd()

# Artifacts base directory
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts", TIMESTAMP)

# Dataset ZIP
ZIP_FILE_NAME = "dataset.zip"
ZIP_FILE_PATH = os.path.join(ROOT_DIR, ZIP_FILE_NAME)

# Data ingestion artifact folders
DATA_INGESTION_DIR = os.path.join(ARTIFACTS_DIR, "DataIngestionArtifacts")
IMBALANCED_CSV = os.path.join(DATA_INGESTION_DIR, "imbalanced_data.csv")
RAW_CSV = os.path.join(DATA_INGESTION_DIR, "raw_data.csv")

class DataIngestionConfig:
    def __init__(self):
        self.ZIP_FILE_PATH = ZIP_FILE_PATH
        self.DATA_INGESTION_DIR = DATA_INGESTION_DIR
        self.DATA_ARTIFACTS_FILE = IMBALANCED_CSV
        self.NEW_DATA_ARTIFACTS_FILE = RAW_CSV

        # Ensure directories exist
        os.makedirs(self.DATA_INGESTION_DIR, exist_ok=True)

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE 
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS 
        self.LABEL = LABEL
        self.TWEET = TWEET

@dataclass
class ModelTrainerConfig: 
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR) 
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR, X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.ACTIVATION = ACTIVATION
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.RANDOM_STATE = RANDOM_STATE
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.VALIDATION_SPLIT = VALIDATION_SPLIT


@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR,BEST_MODEL_DIR)
        # self.BUCKET_NAME = BUCKET_NAME 
        self.MODEL_NAME = MODEL_NAME 



@dataclass
class ModelPusherConfig:

    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        # self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = MODEL_NAME
    