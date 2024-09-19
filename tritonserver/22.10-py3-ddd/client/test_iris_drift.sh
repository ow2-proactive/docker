echo --- DRIFT OFF ---
# ./triton_iris_classification_client_grpc.py "localhost:18001" --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client_http.py "localhost:18000" --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client_http.py "localhost:18080" --model-name iris-classification-model-ddd --mode predict-drift
./triton_iris_classification_client_http.py "localhost:19090" --model-name iris-classification-model-ddd --mode predict-drift
# ./triton_iris_classification_client_http.py "localhost:9090" --model-name iris-classification-model-ddd --mode predict-drift
echo --- DRIFT ON ---
# ./triton_iris_classification_client_grpc.py "localhost:18001" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client_http.py "localhost:18000" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client_http.py "localhost:18080" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
./triton_iris_classification_client_http.py "localhost:19090" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
# ./triton_iris_classification_client_http.py "localhost:9090" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
echo --- DONE ---
