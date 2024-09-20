echo --- DRIFT OFF ---
# ./triton_iris_classification_client.py "localhost:18001" --protocol grpc --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client.py "localhost:18000" --protocol http --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client.py "localhost:18080" --protocol http --model-name iris-classification-model-ddd --mode predict-drift
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client.py "localhost:9090" --protocol http --model-name iris-classification-model-ddd --mode predict-drift
echo --- DRIFT ON ---
# ./triton_iris_classification_client.py "localhost:18001" --protocol grpc --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client.py "localhost:18000" --protocol http --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client.py "localhost:18080" --protocol http --model-name iris-classification-model-ddd --mode predict-drift --drift-data
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client.py "localhost:9090" --protocol http --model-name iris-classification-model-ddd --mode predict-drift --drift-data
echo --- DONE ---
