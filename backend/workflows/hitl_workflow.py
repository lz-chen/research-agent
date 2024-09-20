import asyncio
import mlflow
from config import settings
from llama_index.core.workflow import Workflow
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HumanInTheLoopWorkflow(Workflow):
    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop()  # Store the event loop
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        mlflow.set_experiment(self.__class__.__name__)
        # mlflow.config.enable_async_logging()
        # mlflow.tracing.enable()
        mlflow.llama_index.autolog()
        with mlflow.start_run():
            logger.debug(
                f"{self.__class__.__name__}: MLflow tracking URI {mlflow.get_tracking_uri()}"
            )
            logger.debug(
                f"{self.__class__.__name__}: MLflow artifact store {mlflow.get_artifact_uri()}"
            )
            logger.debug(f"{self.__class__.__name__}: MLflow run started")
            try:
                result = await super().run(*args, **kwargs)
                # mlflow.log_param("status", "success")
                logger.debug(f"{self.__class__.__name__}: Workflow succeeded")
                return result
            except Exception as e:
                # Log the exception details to MLflow
                mlflow.log_param("status", "failed")
                mlflow.log_param("error_message", str(e))
                logger.error(f"Workflow failed with exception: {e}")
                raise  # Re-raise the exception after logging
            finally:
                # mlflow.end_run()
                logger.debug(f"{self.__class__.__name__}: MLflow run ended")
