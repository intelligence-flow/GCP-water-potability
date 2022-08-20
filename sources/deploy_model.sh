export MLFLOW_TRACKING_URI=postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres
export MLFLOW_REGISTRY_URI=postgresql+psycopg2://postgres:XXX@35.xxx.xxx.xx:xxxx/postgres

mlflow models serve -m "models:/water_model/production" --no-conda -h x.x.x.x -p 5000
