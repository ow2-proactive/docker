echo --- DRIFT OFF ---
# ./triton_iris_classification_client_grpc.py "localhost:18001" --model-name iris-classification-model --mode predict-label
# ./triton_iris_classification_client_http.py "localhost:18000" --model-name iris-classification-model --mode predict-label
# ./triton_iris_classification_client_http.py "localhost:18080" --model-name iris-classification-model --mode predict-label
./triton_iris_classification_client_http.py "localhost:19090" --model-name iris-classification-model --mode predict-label
# ./triton_iris_classification_client_http.py "localhost:9090" --model-name iris-classification-model --mode predict-label
echo --- DRIFT ON ---
# ./triton_iris_classification_client_grpc.py "localhost:18001" --model-name iris-classification-model --mode predict-label --drift-data
# ./triton_iris_classification_client_http.py "localhost:18000" --model-name iris-classification-model --mode predict-label --drift-data
# ./triton_iris_classification_client_http.py "localhost:18080" --model-name iris-classification-model --mode predict-label --drift-data
./triton_iris_classification_client_http.py "localhost:19090" --model-name iris-classification-model --mode predict-label --drift-data
# ./triton_iris_classification_client_http.py "localhost:9090" --model-name iris-classification-model --mode predict-label --drift-data
echo --- DONE ---
