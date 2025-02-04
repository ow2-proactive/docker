# docker build -t activeeon/mlflow:latest .
# docker push activeeon/mlflow:latest
# docker run -it --rm -p 5000:5000 --name mlflow activeeon/mlflow:latest /bin/bash -c "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root /app/mlartifacts"
# docker run -d --rm -p 5000:5000 --name mlflow activeeon/mlflow:latest /bin/bash -c "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root /app/mlartifacts"
# docker cp mlflow-training.py mlflow:/app/
# docker exec -it mlflow /bin/bash

# https://github.com/bitnami/containers/tree/main/bitnami/mlflow
FROM bitnami/mlflow:latest

# Install SQLite
USER root
RUN mkdir -p /var/lib/apt/lists/partial && apt-get update && apt-get install -y sqlite3 libsqlite3-dev && apt-get clean

RUN python3 -m pip install py4j

# Ensure /app exists and is writable
RUN mkdir -p /app /app/mlartifacts /app/tmp && chmod -R 777 /app

# Switch back to the original user
USER 1001

ENTRYPOINT [ "" ]