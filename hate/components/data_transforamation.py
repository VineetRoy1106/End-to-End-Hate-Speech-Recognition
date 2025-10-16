# import os
# import re
# import sys
# import string
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# from sklearn.model_selection import train_test_split
# from hate.logger.logger import logger
# from hate.exception import CustomException
# from hate.entity.config_entity import DataTransformationConfig
# from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

# class DataTransformation:
#     def __init__(self, data_transformation_config: DataTransformationConfig,
#                  data_ingestion_artifacts: DataIngestionArtifacts):
#         self.data_transformation_config = data_transformation_config
#         self.data_ingestion_artifacts = data_ingestion_artifacts

#         # Ensure the transformation artifact folder exists
#         os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)

#     def imbalance_data_cleaning(self):
#         try:
#             logger.info("Entered into the imbalance_data_cleaning function")

#             # Read the CSV file (not folder)
#             imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalanced_data_file_path)
#             imbalance_data.drop(self.data_transformation_config.ID,
#                                 axis=self.data_transformation_config.AXIS,
#                                 inplace=self.data_transformation_config.INPLACE)

#             logger.info(f"Exited the imbalance_data_cleaning function and returned imbalance data")
#             return imbalance_data
#         except Exception as e:
#             raise CustomException(e, sys) from e

#     def raw_data_cleaning(self):
#         try:
#             logger.info("Entered into the raw_data_cleaning function")

#             # Read the CSV file (not folder)
#             raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
#             raw_data.drop(self.data_transformation_config.DROP_COLUMNS,
#                           axis=self.data_transformation_config.AXIS,
#                           inplace=self.data_transformation_config.INPLACE)

#             # Fix class labels
#             raw_data[self.data_transformation_config.CLASS].replace({0:1, 2:0}, inplace=True)
#             raw_data.rename(columns={self.data_transformation_config.CLASS:
#                                      self.data_transformation_config.LABEL}, inplace=True)

#             logger.info("Exited the raw_data_cleaning function")
#             return raw_data
#         except Exception as e:
#             raise CustomException(e, sys) from e

#     def concat_dataframe(self):
#         try:
#             logger.info("Entered into the concat_dataframe function")

#             df = pd.concat([self.raw_data_cleaning(), self.imbalance_data_cleaning()])
#             logger.info("Exited the concat_dataframe function")
#             return df
#         except Exception as e:
#             raise CustomException(e, sys) from e

#     def concat_data_cleaning(self, words):
#         try:
#             logger.info("Entered into the concat_data_cleaning function")

#             stemmer = nltk.SnowballStemmer("english")
#             stopword = set(stopwords.words('english'))
#             words = str(words).lower()
#             words = re.sub('\[.*?\]', '', words)
#             words = re.sub('https?://\S+|www\.\S+', '', words)
#             words = re.sub('<.*?>+', '', words)
#             words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
#             words = re.sub('\n', '', words)
#             words = re.sub('\w*\d\w*', '', words)
#             words = [word for word in words.split(' ') if word not in stopword]
#             words = [stemmer.stem(word) for word in words]
#             cleaned_text = " ".join(words)

#             logger.info("Exited the concat_data_cleaning function")
#             return cleaned_text
#         except Exception as e:
#             raise CustomException(e, sys) from e

#     def initiate_data_transformation(self) -> DataTransformationArtifacts:
#         try:
#             logger.info("Entered the initiate_data_transformation method of Data transformation class")

#             df = self.concat_dataframe()
#             df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(
#                 self.concat_data_cleaning
#             )

#             # Save CSV as final.csv inside DataTransformationArtifacts folder
#             final_csv_path = os.path.join(
#                 self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,
#                 "final.csv"
#             )
#             df.to_csv(final_csv_path, index=False, header=True)

#             data_transformation_artifact = DataTransformationArtifacts(
#                 transformed_data_path=final_csv_path
#             )

#             logger.info(f"Returning DataTransformationArtifacts: {final_csv_path}")
#             return data_transformation_artifact

#         except Exception as e:
#             raise CustomException(e, sys) from e


import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from hate.logger.logger import logger
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def imbalance_data_cleaning(self):
        try:
            logger.info("Entered into the imbalance_data_cleaning function")
            imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalanced_data_file_path)
            imbalance_data.drop(
                self.data_transformation_config.ID,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )
            logger.info(f"Exiting imbalance_data_cleaning, shape: {imbalance_data.shape}")
            return imbalance_data
        except Exception as e:
            raise CustomException(e, sys) from e

    def raw_data_cleaning(self):
        try:
            logger.info("Entered into the raw_data_cleaning function")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            raw_data.drop(
                self.data_transformation_config.DROP_COLUMNS,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )

            # Adjust class values
            raw_data[self.data_transformation_config.CLASS].replace({0: 1, 2: 0}, inplace=True)
            raw_data.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL}, inplace=True)

            logger.info(f"Exiting raw_data_cleaning, shape: {raw_data.shape}")
            return raw_data
        except Exception as e:
            raise CustomException(e, sys) from e

    def concat_dataframe(self):
        try:
            logger.info("Entered into the concat_dataframe function")
            df = pd.concat([self.raw_data_cleaning(), self.imbalance_data_cleaning()])
            logger.info(f"Exiting concat_dataframe, shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def concat_data_cleaning(self, text):
        try:
            # Minimal logging to avoid flooding
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))

            words = str(text).lower()
            words = re.sub(r'\[.*?\]', '', words)
            words = re.sub(r'https?://\S+|www\.\S+', '', words)
            words = re.sub(r'<.*?>+', '', words)
            words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub(r'\n', '', words)
            words = re.sub(r'\w*\d\w*', '', words)

            words = [word for word in words.split(' ') if word not in stopword]
            words = [stemmer.stem(word) for word in words]
            return " ".join(words)

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logger.info("Entered initiate_data_transformation")

            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)

            # Create artifact directory
            transformation_dir = os.path.join(
                self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR
            )
            os.makedirs(transformation_dir, exist_ok=True)

            # Save final CSV
            final_csv_path = os.path.join(transformation_dir, "final.csv")
            df.to_csv(final_csv_path, index=False, header=True)
            logger.info(f"Transformed data saved at: {final_csv_path}")

            return DataTransformationArtifacts(transformed_data_path=final_csv_path)

        except Exception as e:
            raise CustomException(e, sys) from e
