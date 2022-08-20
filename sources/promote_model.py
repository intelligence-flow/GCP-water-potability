import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")
mlflow.set_registry_uri("postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres")

client = MlflowClient()

models = mlflow.search_runs('1')
models = models[['metrics.presicion_test','run_id']]
models = models.dropna()

best_model = models.sort_values(by = ['metrics.presicion_test'], ascending = False).head(1)

best_model_run_id = best_model.iloc[0,1]

result = mlflow.register_model("runs:/" + best_model_run_id + "/water_model","water_model")

client.transition_model_version_stage(
        name="water_model",
        version=result.version,
        stage="production"
    )
