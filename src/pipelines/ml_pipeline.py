import sys
import warnings
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils.data_split import DataSplitter
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluator
from src.components.model_registry_and_deploy import ModelRegistryAndDeploy
from src.utils.logger import logging
from config.constants import EXPERIMENT_NAME

warnings.filterwarnings("ignore")


class MLPipeline:
    """
    This class orchestrates the entire Machine Learning pipeline
    for Premier League match prediction.
    """

    def __init__(self):
        """
        Initialize the ML Pipeline
        """
        try:
            logging.info("Initializing ML Pipeline...")
            self.df = None
            self.train_df = None
            self.test_df = None
            self.model_dict = None
            self.evaluation_results = None
        except Exception as e:
            logging.error(f"Error occurred during ML Pipeline initialization: {str(e)}")
            raise

    def ingest_data(self):
        """
        Execute the data ingestion step

        Returns:
            pd.DataFrame: Ingested data
        """
        try:
            logging.info("STEP 1: Starting data ingestion...")
            ingestion = DataIngestion()
            self.df = ingestion.fetch_all_data()
            logging.info(f"Data ingestion completed: {len(self.df)} rows ingested.")
            return self.df
        except Exception as e:
            logging.error(f"Error occurred during data ingestion step: {str(e)}")
            raise

    def transform_data(self):
        """
        Execute the data transformation step

        Returns:
            pd.DataFrame: Transformed data
        """
        try:
            if self.df is None:
                logging.warning("Ingested data is None. Running ingestion first...")
                self.ingest_data()

            logging.info("STEP 2: Starting data transformation...")
            transformation = DataTransformation(self.df)
            self.df = transformation.run_data_transformation()
            logging.info(f"Data transformation completed: shape is {self.df.shape}")
            return self.df
        except Exception as e:
            logging.error(f"Error occurred during data transformation step: {str(e)}")
            raise

    def split_data(self):
        """
        Execute the data splitting step

        Returns:
            tuple: (train_df, test_df)
        """
        try:
            if self.df is None:
                logging.warning("Transformed data is None. Running transformation first...")
                self.transform_data()

            logging.info("STEP 3: Starting data splitting (last 3 weeks as test)...")
            splitter = DataSplitter(self.df)
            self.train_df, self.test_df = splitter.split()
            logging.info(f"Split completed: Train={len(self.train_df)} rows, Test={len(self.test_df)} rows")
            return self.train_df, self.test_df
        except Exception as e:
            logging.error(f"Error occurred during data splitting step: {str(e)}")
            raise

    def train_model(self):
        """
        Execute the model training step

        Returns:
            dict: Dictionary with model and training details
        """
        try:
            if self.train_df is None or self.test_df is None:
                logging.warning("Training/Testing data is missing. Running split first...")
                self.split_data()

            logging.info("STEP 4: Starting model training...")
            trainer = ModelTrainer(self.train_df, self.test_df)
            self.model_dict = trainer.train()
            logging.info(f"Model training completed: {self.model_dict['model_name']}")
            logging.info(f"Parameters: {self.model_dict['params']}")
            return self.model_dict
        except Exception as e:
            logging.error(f"Error occurred during model training step: {str(e)}")
            raise

    def evaluate_model(self):
        """
        Execute the model evaluation step with MLflow logging

        Returns:
            dict: Evaluation results
        """
        try:
            if self.model_dict is None:
                logging.warning("No model found to evaluate. Running training first...")
                self.train_model()

            logging.info("STEP 5: Starting model evaluation and MLflow logging...")
            evaluator = ModelEvaluator(
                model_dict=self.model_dict,
                experiment_name=EXPERIMENT_NAME,
                run_name=self.model_dict["model_name"]
            )
            self.evaluation_results = evaluator.run()
            logging.info("Model evaluation completed successfully")
            return self.evaluation_results
        except Exception as e:
            logging.error(f"Error occurred during model evaluation step: {str(e)}")
            raise

    def deploy_model(self):
        """
        Execute the model registry and deployment step

        Returns:
            bool: True if deployment was successful
        """
        try:
            logging.info("STEP 6: Starting model registry and deployment...")
            try:
                registry = ModelRegistryAndDeploy(metric_name="accuracy", higher_is_better=True)
                registry.run_deployment_pipeline()
                logging.info("Model registry and deployment step completed successfully")
                return True
            except Exception as e:
                logging.warning(f"Deployment skipped due to error: {e}")
                return False
        except Exception as e:
            logging.error(f"Unexpected error in deploy_model wrapper: {str(e)}")
            return False

    def run(self):
        """
        Execute the full ML pipeline orchestration
        """
        try:
            logging.info("=" * 60)
            logging.info("Starting ML Pipeline execution...")
            logging.info("=" * 60)

            # Execution Flow
            self.ingest_data()
            self.transform_data()
            self.split_data()
            self.train_model()
            
            # Wrap evaluation in try-except to prevent pipeline from dying on MLflow errors
            try:
                self.evaluate_model()
            except Exception as e:
                logging.warning(f"MLflow logging/evaluation failed: {e}")
            
            self.deploy_model()

            logging.info("=" * 60)
            logging.info("ML Pipeline execution completed successfully!")
            logging.info("=" * 60)

            return {
                "status": "SUCCESS",
                "message": "Full ML Pipeline completed successfully"
            }

        except Exception as e:
            logging.error("=" * 60)
            logging.error("ML Pipeline execution failed!")
            logging.error(f"Error detail: {str(e)}")
            logging.error("=" * 60)
            raise


if __name__ == "__main__":
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        logging.error(f"Scheduled ML pipeline run failed: {str(e)}")
        sys.exit(1)