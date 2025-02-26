echo --- DRIFT OFF ---
# ./triton_iris_classification_client.py "localhost:18001" --protocol grpc --model-name iris-classification-model --mode predict-label
# ./triton_iris_classification_client.py "localhost:18000" --protocol http --model-name iris-classification-model --mode predict-label
# ./triton_iris_classification_client.py "localhost:18080" --protocol http --model-name iris-classification-model --mode predict-label
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model --mode predict-label
echo --- DRIFT ON ---
# ./triton_iris_classification_client.py "localhost:18001" --protocol grpc --model-name iris-classification-model --mode predict-label --drift-data
# ./triton_iris_classification_client.py "localhost:18000" --protocol http --model-name iris-classification-model --mode predict-label --drift-data
# ./triton_iris_classification_client.py "localhost:18080" --protocol http --model-name iris-classification-model --mode predict-label --drift-data
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model --mode predict-label --drift-data
echo --- DONE ---
sleep 1
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model2 --mode predict-label
./triton_iris_classification_client.py "localhost:19090" --protocol http --model-name iris-classification-model2 --mode predict-label --drift-data
echo --- DONE ---
