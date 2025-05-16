from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import (
    DataIngestionTrainingPipeline,
)
from src.datascience.pipeline.data_validation_pipeline import (
    DataValidationTrainingPipeline,
)
from src.datascience.pipeline.data_transformation_pipeline import (
    DataTransformationPipeline,
)
from src.datascience.pipeline.model_trainer_pipeline import (
    ModelTrainerTrainingPipeline,
)
from src.datascience.pipeline.model_evaluation_pipeline import (
    ModelEvaluationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.initiate_data_validation()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = DataTransformationPipeline()
    obj.initiate_data_transformation()

    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")
except Exception as e:
    raise e


STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.initiate_model_training()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")

except Exception as e:
    raise e


STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.initiate_model_evaluation()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")

except Exception as e:
    raise e
