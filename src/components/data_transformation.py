import pandas as pd
import sys
from src.utils.logger import logging
from src.utils.exception import MyException
from src.feature_engineering.feature_enginnering import FeatureEngineering

class DataTransformation:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def run_data_transformation(self):        
        try:
            logging.info("Start the Data Transformation Pipeline")
            
            transformation = FeatureEngineering(self.df)
            transformed_df = transformation.run()

            logging.info(f"Transformed data had {len(transformed_df)} samples and {len(transformed_df.columns)} Columns")
            logging.info("End the Data Transformation Pipeline")
            return transformed_df
        
        except Exception as e:
            logging.error(f"Error during DataTransformation Pipeline: {str(e)}")
            raise MyException(e, sys)
