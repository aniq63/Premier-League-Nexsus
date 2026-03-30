import os
import sys
import io
import logging as std_logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.utils.logger import logging
from src.utils.exception import MyException

MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


# ============================================================
# ModelEvaluator Class
# ============================================================

class ModelEvaluator:
    """
    Evaluate a trained model returned from ModelTrainer.train() and log
    everything to MLflow: metrics, params, model artifact, plots, and data sample.

    Args:
        model_dict (dict)      : Output from ModelTrainer.train()
        experiment_name (str)  : MLflow experiment name to log into
        run_name (str | None)  : Optional MLflow run name
    """

    def __init__(
        self,
        model_dict: dict,
        experiment_name: str = "EPL_Model_Evaluation",
        run_name: str = None,
    ):
        try:
            self.model_dict      = model_dict
            self.model           = model_dict["model"]
            self.X_test          = model_dict["X_test"]        # pd.DataFrame
            self.y_test          = model_dict["y_test"]        # np.ndarray (encoded)
            self.le              = model_dict["label_encoder"]
            self.model_name      = model_dict["model_name"]
            self.params          = model_dict["params"]
            self.input_features  = self.params.get(
                "input_features", list(self.X_test.columns)
            )
            self.experiment_name = experiment_name
            self.run_name        = run_name or self.model_name
            self._results        = {}           # populated by evaluate()

            logging.info(
                f"ModelEvaluator initialized | model={self.model_name} | "
                f"test_size={len(self.X_test)} rows | "
                f"features={len(self.input_features)}"
            )
        except KeyError as ke:
            logging.error(f"model_dict is missing key: {ke}")
            raise MyException(ke, sys)
        except Exception as e:
            logging.error(f"ModelEvaluator init failed: {e}")
            raise MyException(e, sys)

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

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

    def _compute_metrics(self, y_pred: np.ndarray) -> dict:
        """Compute and return all classification metrics."""
        metrics = {
            "accuracy":  accuracy_score(self.y_test, y_pred),
            "precision": precision_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall":    recall_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1_score":  f1_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
        }
        # Per-class metrics
        labels = list(range(len(self.le.classes_)))
        for i, cls in enumerate(self.le.classes_):
            metrics[f"precision_{cls}"] = precision_score(
                self.y_test, y_pred, labels=labels, average=None, zero_division=0
            )[i]
            metrics[f"recall_{cls}"] = recall_score(
                self.y_test, y_pred, labels=labels, average=None, zero_division=0
            )[i]
            metrics[f"f1_{cls}"] = f1_score(
                self.y_test, y_pred, labels=labels, average=None, zero_division=0
            )[i]
        return metrics

    def _compute_feature_importance(self) -> pd.Series:
        """Extract feature importances from model."""
        if not hasattr(self.model, "feature_importances_"):
            logging.warning("Model does not expose feature_importances_; skipping.")
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.feature_importances_, index=self.input_features
        ).sort_values(ascending=False)

    def _build_evaluation_figure(
        self,
        cm: np.ndarray,
        feat_imp: pd.Series,
    ) -> plt.Figure:
        """Build a single 2-panel figure: confusion matrix + top feature importances."""
        top_n = min(20, len(feat_imp))
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f"{self.model_name} – Test Set Evaluation",
            fontsize=15, fontweight="bold", y=1.01
        )

        # Panel 1 – Confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.le.classes_
        )
        disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")

        # Panel 2 – Top N feature importances
        top_features = feat_imp.head(top_n)
        top_features.sort_values().plot(
            kind="barh", ax=axes[1], color="steelblue", edgecolor="white"
        )
        axes[1].set_title(
            f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Importance")
        axes[1].tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        return fig

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Run model inference on the test set and compute all metrics and
        feature importances.

        Returns:
            dict with keys: metrics, confusion_matrix, classification_report,
                            feature_importances, y_pred, y_pred_proba
        """
        try:
            logging.info("=" * 60)
            logging.info(f"MODEL EVALUATION START | {self.model_name}")
            logging.info("=" * 60)

            logging.info(f"Running inference on {len(self.X_test)} test samples...")
            y_pred       = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)

            # Metrics
            logging.info("Computing classification metrics...")
            metrics = self._compute_metrics(y_pred)

            logging.info(
                f"Evaluation complete | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1_score']:.4f}"
            )

            # Confusion matrix & report
            cm     = confusion_matrix(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred,
                target_names=list(self.le.classes_),
                zero_division=0,
            )
            logging.info(f"Classification Report:\n{report}")

            # Feature importances
            feat_imp = self._compute_feature_importance()

            self._results = {
                "metrics":               metrics,
                "confusion_matrix":      cm,
                "classification_report": report,
                "feature_importances":   feat_imp,
                "y_pred":                y_pred,
                "y_pred_proba":          y_pred_proba,
            }

            logging.info("=" * 60)
            logging.info("MODEL EVALUATION COMPLETE")
            logging.info("=" * 60)

            return self._results

        except AttributeError as ae:
            logging.error(f"Model is not fitted or attribute missing: {ae}")
            raise MyException(ae, sys)
        except Exception as e:
            logging.error(f"Unexpected error during evaluation: {e}")
            raise MyException(e, sys)

    def log_to_mlflow(self) -> None:
        """
        Log everything to MLflow:
          - Params  : all model + search params from model_dict
          - Metrics : accuracy, precision, recall, f1 (overall + per-class)
          - Tags    : model name, feature count, test size
          - Artifacts:
              * model itself (mlflow.sklearn)
              * confusion matrix + feature importance figure (PNG)
              * full classification report (TXT)
              * feature importances table (CSV)
              * test data sample (CSV)
        """
        if not self._results:
            raise RuntimeError(
                "Call .evaluate() before .log_to_mlflow()."
            )

        try:
            logging.info(f"Logging to MLflow | experiment='{self.experiment_name}'")
            
            # Connect to MLflow
            self._connect_to_mlflow()
            
            mlflow.set_experiment(self.experiment_name)
            
            # Silence internal MLflow warnings that standard warning filters might miss
            std_logging.getLogger("mlflow").setLevel(std_logging.ERROR)
            
            with mlflow.start_run(run_name=self.run_name):

                # ── Tags ──────────────────────────────────────────────────
                mlflow.set_tags({
                    "model_name":    self.model_name,
                    "feature_count": len(self.input_features),
                    "test_size":     len(self.X_test),
                })

                # ── Params ────────────────────────────────────────────────
                logging.info("Logging model params to MLflow...")
                for key, val in self.params.items():
                    if key == "input_features":
                        # Too long for a param; log as artifact instead
                        continue
                    mlflow.log_param(key, val)

                # ── Metrics ───────────────────────────────────────────────
                logging.info("Logging metrics to MLflow...")
                for metric_name, metric_val in self._results["metrics"].items():
                    mlflow.log_metric(metric_name, round(float(metric_val), 6))

                # ── Model artifact (sklearn flavour) ──────────────────────
                logging.info("Logging model artifact to MLflow...")
                import warnings
                with warnings.catch_warnings():
                    # Suppress the artifact_path deprecation and pickle security warnings
                    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
                    mlflow.sklearn.log_model(
                        sk_model=self.model,
                        name="model",  # Use name instead of artifact_path to fix MLflow 2.x warning
                        registered_model_name=self.model_name,
                    )

                # ── Evaluation figure ─────────────────────────────────────
                logging.info("Generating and logging evaluation figure...")
                fig = self._build_evaluation_figure(
                    cm=self._results["confusion_matrix"],
                    feat_imp=self._results["feature_importances"],
                )
                mlflow.log_figure(fig, artifact_file="evaluation_plots.png")
                plt.close(fig)

                # ── Classification report (text) ──────────────────────────
                report_buf = io.StringIO()
                report_buf.write(self._results["classification_report"])
                mlflow.log_text(
                    report_buf.getvalue(),
                    artifact_file="classification_report.txt",
                )

                # ── Feature importances (CSV) ─────────────────────────────
                feat_imp = self._results["feature_importances"]
                if not feat_imp.empty:
                    fi_df = feat_imp.reset_index()
                    fi_df.columns = ["feature", "importance"]
                    fi_buf = io.StringIO()
                    fi_df.to_csv(fi_buf, index=False)
                    mlflow.log_text(
                        fi_buf.getvalue(),
                        artifact_file="feature_importances.csv",
                    )

                # ── Input features list (text) ────────────────────────────
                mlflow.log_text(
                    "\n".join(self.input_features),
                    artifact_file="input_features.txt",
                )

                # ── Test data sample (CSV) ────────────────────────────────
                sample_buf = io.StringIO()
                test_sample = self.X_test.copy()
                test_sample["y_true_encoded"] = self.y_test
                test_sample["y_true_label"]   = self.le.inverse_transform(self.y_test)
                test_sample["y_pred_encoded"] = self._results["y_pred"]
                test_sample["y_pred_label"]   = self.le.inverse_transform(
                    self._results["y_pred"]
                )
                test_sample.head(50).to_csv(sample_buf, index=False)
                mlflow.log_text(
                    sample_buf.getvalue(),
                    artifact_file="test_sample_predictions.csv",
                )

                run_id = mlflow.active_run().info.run_id
                logging.info(
                    f"MLflow logging complete | run_id={run_id} | "
                    f"experiment='{self.experiment_name}'"
                )

        except Exception as e:
            logging.error(f"MLflow logging failed: {e}")
            raise MyException(e, sys)

    def run(self) -> dict:
        """
        Convenience method: evaluate() then log_to_mlflow() in one call.

        Returns:
            Evaluation results dict (same as evaluate()).
        """
        results = self.evaluate()
        self.log_to_mlflow()
        return results