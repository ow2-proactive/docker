echo --- DRIFT OFF ---
./triton_iris_classification_client.py "localhost:18001" --model-name iris-classification-model --mode predict-label
echo --- DRIFT ON ---
./triton_iris_classification_client.py "localhost:18001" --model-name iris-classification-model --mode predict-label --drift-data
echo --- DONE ---
