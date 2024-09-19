echo --- DRIFT OFF ---
./triton_iris_classification_client.py "localhost:18001" --model-name iris-classification-model-ddd --mode predict-drift
echo --- DRIFT ON ---
./triton_iris_classification_client.py "localhost:18001" --model-name iris-classification-model-ddd --mode predict-drift --drift-data
echo --- DONE ---
