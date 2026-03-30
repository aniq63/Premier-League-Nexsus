import os
import sys
import json
import pickle
import tempfile
import boto3
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient


from config.constants import EXPERIMENT_NAME, MODEL_NAME
from src.utils.logger import logging
from src.utils.exception import MyException

load_dotenv()

MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

AWS_ACCESS_KEY_ID        = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY    = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION               = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME           = os.getenv("S3_BUCKET_NAME")


class ModelRegistryAndDeploy:
    """
    Picks the best run from MLflow, promotes it through Staging → Production,
    and uploads the final model to S3.

    NOTE: ModelEvaluator already calls mlflow.sklearn.log_model() with
    registered_model_name set, so by the time this class runs the model
    is already registered. We only need to:
        1. Find which registered version has the best metric
        2. Move it through Staging → Production
        3. Upload the pickle to S3
    """

    def __init__(self, metric_name: str = "accuracy", higher_is_better: bool = True):
        self.experiment_name    = EXPERIMENT_NAME
        self.model_name         = MODEL_NAME
        self.metric_name        = metric_name
        self.higher_is_better   = higher_is_better
        self.best_artifact_path = "model"

        try:
            logging.info("Initializing ModelRegistryAndDeploy...")
            self._connect_to_mlflow()
            self.client = MlflowClient()
        except Exception as e:
            logging.error(f"Failed during initialization: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def _connect_to_mlflow(self):
        try:
            if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
                os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
                os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    
            uri = (
                MLFLOW_TRACKING_URI
                if MLFLOW_TRACKING_URI and MLFLOW_TRACKING_URI.startswith("http")
                else "https://dagshub.com/aniqramzan5758/EPL_Match_Prediction.mlflow"
            )
    
            mlflow.set_tracking_uri(uri)
            logging.info(f"Connected to MLflow at: {uri}")
    
        except Exception as e:
            logging.error(f"Error configuring MLflow: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def get_best_run(self):
        """
        Find the best finished run that has a logged model artifact.

        Uses 3 strategies so we never get a false 'not found':
            A) Read the mlflow.log-model.history tag  (fast, no extra HTTP)
            B) List artifacts directly                (one HTTP call per run)
            C) Fall back to the Model Registry        (always works if evaluator ran)
        """
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(
                    f"Experiment '{self.experiment_name}' not found.\n"
                    f"Make sure EXPERIMENT_NAME in config/constants.py matches "
                    f"exactly what ModelEvaluator uses."
                )

            experiment_id = experiment.experiment_id
            logging.info(f"Searching experiment '{self.experiment_name}' (id={experiment_id})")

            order_by = (
                f"metrics.{self.metric_name} "
                f"{'DESC' if self.higher_is_better else 'ASC'}"
            )
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=[order_by],
                max_results=20,
            )

            if not runs:
                raise ValueError(f"No finished runs in experiment '{self.experiment_name}'.")

            logging.info(f"Found {len(runs)} finished run(s). Scanning for model artifact...")

            for run in runs:
                run_id     = run.info.run_id
                metric_val = run.data.metrics.get(self.metric_name, "N/A")

                # Strategy A: check the model-log history tag (no extra HTTP call)
                history_str = run.data.tags.get("mlflow.log-model.history", "")
                if history_str:
                    try:
                        history = json.loads(history_str)
                        if history:
                            artifact_path = history[0].get("artifact_path", "model")
                            logging.info(
                                f"  ✓ run {run_id[:8]}... | "
                                f"{self.metric_name}={metric_val} | "
                                f"path='{artifact_path}' [tag]"
                            )
                            self.best_artifact_path = artifact_path
                            return run
                    except Exception:
                        pass    # malformed tag — fall through to B

                # Strategy B: list artifacts directly (one HTTP call)
                try:
                    listed = [a.path for a in self.client.list_artifacts(run_id)]
                    logging.info(f"  run {run_id[:8]}... artifacts: {listed}")
                    hit = next(
                        (p for p in listed if p == "model" or p.startswith("model")),
                        None,
                    )
                    if hit:
                        logging.info(
                            f"  ✓ run {run_id[:8]}... | "
                            f"{self.metric_name}={metric_val} | "
                            f"path='{hit}' [listing]"
                        )
                        self.best_artifact_path = hit
                        return run
                    else:
                        logging.info(f"  ✗ run {run_id[:8]}... — no model artifact, skipping.")
                except Exception as list_err:
                    logging.warning(f"  Could not list artifacts for {run_id[:8]}: {list_err}")

            # Strategy C: fall back to the Model Registry
            # ModelEvaluator already registered the model via registered_model_name=,
            # so this is the reliable safety net.
            logging.warning(
                "No model artifact found by scanning runs. "
                "Falling back to Model Registry..."
            )
            all_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if not all_versions:
                raise ValueError(
                    f"Model '{self.model_name}' has no registered versions.\n"
                    f"Make sure ModelEvaluator.log_to_mlflow() completed successfully."
                )

            best_version, best_metric = None, None
            for v in all_versions:
                try:
                    r = self.client.get_run(v.run_id)
                    m = r.data.metrics.get(self.metric_name)
                    if m is None:
                        continue
                    if (best_metric is None
                            or (self.higher_is_better and m > best_metric)
                            or (not self.higher_is_better and m < best_metric)):
                        best_metric, best_version = m, v
                except Exception:
                    continue

            if best_version is None:
                raise ValueError(
                    f"No registered version of '{self.model_name}' has "
                    f"metric '{self.metric_name}'."
                )

            self.best_artifact_path = "model"
            logging.info(
                f"  ✓ Found via registry | run={best_version.run_id[:8]}... | "
                f"{self.metric_name}={best_metric}"
            )
            return self.client.get_run(best_version.run_id)

        except Exception as e:
            logging.error(f"Failed to fetch best run: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def register_model(self, run_id: str) -> int:
        """
        Register the model.
        If ModelEvaluator already registered it for this run, we find that
        existing version instead of failing.
        """
        try:
            artifact_path = getattr(self, "best_artifact_path", "model")
            model_uri     = f"runs:/{run_id}/{artifact_path}"
            logging.info(
                f"Registering model from run {run_id[:8]}... "
                f"path='{artifact_path}' as '{self.model_name}'"
            )
            registered = mlflow.register_model(model_uri=model_uri, name=self.model_name)
            logging.info(f"Model registered — Version {registered.version}")
            return int(registered.version)

        except Exception as e:
            # Recover: find the version that was already registered for this run
            logging.warning(
                f"register_model raised: {e}\n"
                f"Looking for existing version for run {run_id[:8]}..."
            )
            try:
                all_versions = self.client.search_model_versions(
                    f"name='{self.model_name}'"
                )
                for v in all_versions:
                    if v.run_id == run_id:
                        logging.info(
                            f"Found existing version {v.version} for run {run_id[:8]}"
                        )
                        return int(v.version)
                raise ValueError(
                    f"run_id {run_id} has no registered version for '{self.model_name}'."
                )
            except Exception as inner_e:
                logging.error(f"Failed to recover registered version: {inner_e}")
                raise MyException(inner_e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def move_to_staging(self, version: int):
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=str(version),
                stage="Staging",
                archive_existing_versions=False,
            )
            logging.info(f"Model '{self.model_name}' v{version} → STAGING")
        except Exception as e:
            logging.error(f"Failed to move to STAGING: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def compare_staging_vs_production(self) -> bool:
        try:
            def get_metric_for_stage(stage):
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
                if not versions:
                    return None, None
                latest = versions[0]
                run    = self.client.get_run(latest.run_id)
                return run.data.metrics.get(self.metric_name), latest.version

            staging_metric,    staging_ver    = get_metric_for_stage("Staging")
            production_metric, production_ver = get_metric_for_stage("Production")

            logging.info(f"Metric comparison ({self.metric_name}):")
            logging.info(f"  STAGING    v{staging_ver}:    {staging_metric}")
            logging.info(f"  PRODUCTION v{production_ver}: {production_metric}")

            if production_metric is None:
                logging.warning("No PRODUCTION model yet — promoting STAGING automatically.")
                return True

            if staging_metric is None:
                logging.warning("STAGING metric unreadable — skipping promotion.")
                return False

            staging_wins = (
                staging_metric > production_metric
                if self.higher_is_better
                else staging_metric < production_metric
            )
            logging.info(
                "STAGING is better → will promote."
                if staging_wins
                else "PRODUCTION is still better → keeping it."
            )
            return staging_wins

        except Exception as e:
            logging.error(f"Comparison failed: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def promote_to_production(self):
        try:
            staging = self.client.get_latest_versions(self.model_name, stages=["Staging"])
            if not staging:
                logging.warning("No STAGING version to promote.")
                return

            version = staging[0].version
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info(f"Model '{self.model_name}' v{version} → PRODUCTION")
        except Exception as e:
            logging.error(f"Failed to promote to PRODUCTION: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def upload_model_to_s3(self, stage: str = "Production"):
        try:
            if not S3_BUCKET_NAME or not AWS_ACCESS_KEY_ID:
                logging.warning("S3 credentials missing — skipping upload.")
                return None

            s3 = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
            )

            versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            if not versions:
                logging.warning(f"No model in '{stage}' stage — skipping S3 upload.")
                return None

            version   = versions[0]
            model_uri = f"models:/{self.model_name}/{stage}"
            logging.info(f"Loading model from MLflow ({model_uri})...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                model     = mlflow.sklearn.load_model(model_uri)
                local_pkl = os.path.join(tmp_dir, "model.pkl")
                with open(local_pkl, "wb") as f:
                    pickle.dump(model, f)

                s3_key = f"models/{self.model_name}/{stage}/model.pkl"
                logging.info(f"Uploading to s3://{S3_BUCKET_NAME}/{s3_key} ...")
                s3.upload_file(local_pkl, S3_BUCKET_NAME, s3_key)

                metadata = {
                    "model_name": self.model_name,
                    "stage":      stage,
                    "version":    version.version,
                    "run_id":     version.run_id,
                    "s3_bucket":  S3_BUCKET_NAME,
                    "s3_key":     s3_key,
                }
                meta_key = f"models/{self.model_name}/{stage}/metadata.json"
                s3.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=meta_key,
                    Body=json.dumps(metadata, indent=2),
                )
                logging.info(f"Metadata → s3://{S3_BUCKET_NAME}/{meta_key}")

            logging.info("S3 upload complete.")
            return s3_key

        except Exception as e:
            logging.error(f"S3 upload failed: {e}")
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def run_deployment_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("MODEL REGISTRY & DEPLOYMENT PIPELINE START")
            logging.info("=" * 60)

            best_run = self.get_best_run()
            run_id   = best_run.info.run_id

            version = self.register_model(run_id)
            self.move_to_staging(version)

            if self.compare_staging_vs_production():
                self.promote_to_production()
                self.upload_model_to_s3(stage="Production")
            else:
                logging.info("Production model retained. S3 upload skipped.")

            logging.info("=" * 60)
            logging.info("MODEL REGISTRY & DEPLOYMENT PIPELINE COMPLETE")
            logging.info("=" * 60)

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise MyException(e, sys)