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


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = DataTransformationPipeline()
        obj.initiate_data_transformation()

        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n")
    except Exception as e:
        raise e
