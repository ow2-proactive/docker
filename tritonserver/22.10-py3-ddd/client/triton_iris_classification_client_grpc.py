#!/usr/bin/env python3

import sys
import typer
import numpy as np
import logging
from typing import Tuple, Optional, List

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = typer.Typer()


def get_triton_client(server_url: str) -> grpcclient.InferenceServerClient:
    """
    Initialize the Triton inference server client.
    """
    try:
        triton_client = grpcclient.InferenceServerClient(url=server_url)
        return triton_client
    except Exception as e:
        logging.error(f"Channel creation failed: {str(e)}")
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
        # True label is [0] or [1] depending on drift_data
        true_label = np.array([0]) if not drift_data else np.array([1])

    logging.info(f"True label:\n{true_label}")

    if drift_data:
        noise_std = 10.0
        noise = np.random.normal(0, noise_std, input_data.shape)
        input_data += noise
        logging.info(f"Drifted input data:\n{input_data}")

    return input_data, true_label


def prepare_inference_inputs(input_data: np.ndarray) -> List[grpcclient.InferInput]:
    """
    Prepare the input tensors for the inference request.
    """
    batch_size = input_data.shape[0]
    inputs = [grpcclient.InferInput('float_input', [batch_size, input_data.shape[1]], "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    logging.info(f"Prepared input data (float_input):\n{inputs[0]}")
    return inputs


def prepare_inference_outputs(mode: str) -> List[grpcclient.InferRequestedOutput]:
    """
    Prepare the output tensors for the inference request.
    """
    outputs = [grpcclient.InferRequestedOutput('label')]
    if mode == "predict-label":
        outputs.append(grpcclient.InferRequestedOutput('probabilities'))
    return outputs


def perform_inference(
    triton_client: grpcclient.InferenceServerClient,
    model_name: str,
    inputs: List[grpcclient.InferInput],
    outputs: List[grpcclient.InferRequestedOutput],
    model_version: str
) -> grpcclient.InferResult:
    """
    Perform the inference request.
    """
    try:
        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version
        )
        return results
    except InferenceServerException as e:
        logging.error(f"Inference failed: {str(e)}")
        sys.exit(1)


def check_inference_statistics(
    triton_client: grpcclient.InferenceServerClient,
    model_name: str
) -> None:
    """
    Check the inference statistics for the model.
    """
    try:
        statistics = triton_client.get_inference_statistics(model_name=model_name)
        if len(statistics.model_stats) != 1:
            logging.error("Inference Statistics check failed")
            sys.exit(1)
    except InferenceServerException as e:
        logging.error(f"Failed to retrieve inference statistics: {str(e)}")
        sys.exit(1)


def get_output_data(
    results: grpcclient.InferResult,
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
    model_version: str = typer.Option("", help="Version of the model"),
    drift_data: bool = typer.Option(False, help="Flag to indicate whether drift data should be used"),
    mode: str = typer.Option("predict-label", help="Mode of operation: 'predict-label' or 'predict-drift'")
):
    """
    Main function to run the inference client.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Get Triton client
    triton_client = get_triton_client(server_url)

    # Load data
    input_data, true_label = load_data(drift_data, mode)

    # Prepare inputs and outputs
    inputs = prepare_inference_inputs(input_data)
    outputs = prepare_inference_outputs(mode)

    # Perform inference
    results = perform_inference(triton_client, model_name, inputs, outputs, model_version)

    # Check inference statistics
    check_inference_statistics(triton_client, model_name)

    # Get output data
    output0_data, output1_data = get_output_data(results, mode)

    # Calculate accuracy
    if true_label is not None:
        calculate_accuracy(true_label, output0_data)

    logging.info(f'PASS: {model_name}')


if __name__ == "__main__":
    app()
