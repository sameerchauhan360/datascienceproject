import pandas as pd
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import os
import joblib
from urllib.parse import urlparse
from pathlib import Path
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import read_yaml, create_directories, save_json
import os

# os.environ["MLFLOW_TRACKING_URI"] = (
#     "https://dagshub.com/sameerchauhan360/datascienceproject.mlflow"
# )
# os.environ["MLFLOW_TRACKING_USERNAME"] = "sameerchauhan360"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "90ff337d729076a2440f8367026ab6ff75e1300a"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(actual=test_y, pred=predicted_qualities)

            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(data=scores, path=Path(self.config.metric_file_name))

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="ElasticnetModel"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
