#!/usr/bin/env python3

"""
Triton Iris Classification Client Script

This script implements a client for the Iris Classification model deployed on the Triton Inference Server.
It demonstrates how to perform inference requests, handle different protocols (HTTP/gRPC), and process
both normal and drift data for the Iris dataset.

Key Features:
1. Supports both HTTP and gRPC protocols for communication with Triton Inference Server.
2. Handles inference requests for Iris classification and drift detection.
3. Loads and preprocesses Iris dataset, including options for introducing drift.
4. Calculates and logs model accuracy.
5. Flexible configuration options for server URL, model name, version, and operation mode.

Main Components:
- Triton Client: Communicates with the Triton Inference Server using either HTTP or gRPC.
- Data Loading: Prepares Iris dataset for inference, with options for introducing drift.
- Inference Preparation: Formats input data and output requests for Triton server.
- Results Processing: Handles inference results and calculates accuracy.

Usage:
Run the script with the following command:
    python triton_iris_classification_client.py [SERVER_URL] [OPTIONS]

Arguments:
    SERVER_URL  URL of the Triton server (required)

Options:
    --model-name TEXT      Name of the model (default: "iris-classification-model")
    --model-version INT    Version of the model (default: 1)
    --drift-data BOOLEAN   Flag to indicate whether drift data should be used (default: False)
    --mode TEXT            Mode of operation: 'predict-label' or 'predict-drift' (default: "predict-label")
    --protocol TEXT        Protocol to use: 'http' or 'grpc' (default: "http")

The script will then:
1. Connect to the specified Triton Inference Server.
2. Load and preprocess the Iris dataset (with or without drift, based on options).
3. Perform inference using the specified model and protocol.
4. Process the results and calculate accuracy (if applicable).
5. Log the results and performance metrics.

Note: This script assumes that the Iris classification model is properly deployed on the Triton server.
For drift detection, it assumes a corresponding drift detection model is available.

For more detailed information, refer to the function docstrings within the script.
"""

import sys
import typer
import logging
import numpy as np

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple, Optional, List, Union

app = typer.Typer()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_triton_client(server_url: str, protocol: str) -> Union[httpclient.InferenceServerClient, grpcclient.InferenceServerClient]:
    """
    Initialize the Triton inference server client.
    """
    try:
        if protocol == "http":
            return httpclient.InferenceServerClient(url=server_url)
        elif protocol == "grpc":
            return grpcclient.InferenceServerClient(url=server_url)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    except Exception as e:
        logging.error(f"Client creation failed: {str(e)}")
        sys.exit(1)


def load_data(drift_data: bool, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load the Iris dataset and prepare the input data and true labels.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    input_data = X_test.astype(np.float32)
    logging.info(f"Input data:\n{input_data}")
    true_label = None
    if mode == "predict-label":
        true_label = y_test
    elif mode == "predict-drift":
        true_label = np.array([0]) if not drift_data else np.array([1])
    logging.info(f"True label:\n{true_label}")
    if drift_data:
        noise_std = 10.0
        noise = np.random.normal(0, noise_std, input_data.shape)
        input_data += noise
        logging.info(f"Drifted input data:\n{input_data}")
    return input_data, true_label


def prepare_inference_inputs(input_data: np.ndarray, protocol: str) -> Union[List[httpclient.InferInput], List[grpcclient.InferInput]]:
    """
    Prepare the input tensors for the inference request.
    """
    batch_size = input_data.shape[0]
    if protocol == "http":
        inputs = [httpclient.InferInput('float_input', [batch_size, input_data.shape[1]], np_to_triton_dtype(input_data.dtype))]
    elif protocol == "grpc":
        inputs = [grpcclient.InferInput('float_input', [batch_size, input_data.shape[1]], np_to_triton_dtype(input_data.dtype))]
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    inputs[0].set_data_from_numpy(input_data)
    logging.info(f"Prepared input data (float_input):\n{inputs[0]}")
    return inputs


def prepare_inference_outputs(mode: str, protocol: str) -> Union[List[httpclient.InferRequestedOutput], List[grpcclient.InferRequestedOutput]]:
    """
    Prepare the output tensors for the inference request.
    """
    if protocol == "http":
        outputs = [httpclient.InferRequestedOutput('label')]
        if mode == "predict-label":
            outputs.append(httpclient.InferRequestedOutput('probabilities'))
    elif protocol == "grpc":
        outputs = [grpcclient.InferRequestedOutput('label')]
        if mode == "predict-label":
            outputs.append(grpcclient.InferRequestedOutput('probabilities'))
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    return outputs


def perform_inference(
    triton_client: Union[httpclient.InferenceServerClient, grpcclient.InferenceServerClient],
    model_name: str,
    inputs: Union[List[httpclient.InferInput], List[grpcclient.InferInput]],
    outputs: Union[List[httpclient.InferRequestedOutput], List[grpcclient.InferRequestedOutput]],
    model_version: int
) -> Union[httpclient.InferResult, grpcclient.InferResult]:
    """
    Perform the inference request.
    """
    try:
        results = triton_client.infer(
            model_name=model_name,
            model_version=str(model_version),
            inputs=inputs,
            outputs=outputs
        )
        return results
    except InferenceServerException as e:
        logging.error(f"Inference failed: {str(e)}")
        sys.exit(1)


def check_inference_statistics(
    triton_client: Union[httpclient.InferenceServerClient, grpcclient.InferenceServerClient],
    model_name: str,
    protocol: str
) -> None:
    """
    Check the inference statistics for the model.
    """
    try:
        statistics = triton_client.get_inference_statistics(model_name=model_name)
        if protocol == "http":
            if len(statistics['model_stats']) != 1:
                logging.error("Inference Statistics check failed")
                sys.exit(1)
        elif protocol == "grpc":
            if len(statistics.model_stats) != 1:
                logging.error("Inference Statistics check failed")
                sys.exit(1)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    except InferenceServerException as e:
        logging.error(f"Failed to retrieve inference statistics: {str(e)}")
        sys.exit(1)


def get_output_data(
    results: Union[httpclient.InferResult, grpcclient.InferResult],
    mode: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Retrieve and log the output data from the inference results.
    """
    output0_data = results.as_numpy('label')
    logging.info(f"Output data (label):\n{output0_data}")
    output1_data = None
    if mode == "predict-label":
        output1_data = results.as_numpy('probabilities')
        logging.info(f"Output data (probabilities):\n{output1_data}")
    return output0_data, output1_data


def calculate_accuracy(true_label: np.ndarray, predicted_label: np.ndarray) -> float:
    """
    Calculate and log the model accuracy.
    """
    accuracy = accuracy_score(true_label, predicted_label)
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")
    return accuracy


@app.command()
def main(
    server_url: str = typer.Argument(..., help="URL of the Triton server"),
    model_name: str = typer.Option("iris-classification-model", help="Name of the model"),
    model_version: int = typer.Option(1, help="Version of the model"),
    drift_data: bool = typer.Option(False, help="Flag to indicate whether drift data should be used"),
    mode: str = typer.Option("predict-label", help="Mode of operation: 'predict-label' or 'predict-drift'"),
    protocol: str = typer.Option("http", help="Protocol to use: 'http' or 'grpc'")
):
    """
    Main function to run the inference client.
    """
    # Get Triton client
    triton_client = get_triton_client(server_url, protocol)
    # Load data
    input_data, true_label = load_data(drift_data, mode)
    # Prepare inputs and outputs
    inputs = prepare_inference_inputs(input_data, protocol)
    outputs = prepare_inference_outputs(mode, protocol)
    # Perform inference
    results = perform_inference(triton_client, model_name, inputs, outputs, model_version)
    # Check inference statistics
    # check_inference_statistics(triton_client, model_name, protocol)
    # Get output data
    output0_data, output1_data = get_output_data(results, mode)
    # Calculate accuracy
    if true_label is not None:
        calculate_accuracy(true_label, output0_data)
    logging.info(f'PASS: {model_name}')


if __name__ == "__main__":
    app()
