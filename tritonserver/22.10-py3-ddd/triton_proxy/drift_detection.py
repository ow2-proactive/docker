#!/usr/bin/env python3

"""
Drift Detection Script for Triton Inference Server

This script implements a drift detection mechanism for models deployed on the Triton Inference Server.
It listens for updates from a Redis pub/sub channel, processes incoming inference data, predicts drift
using a dedicated drift detection model, and records drift occurrences and statistics.

Key Features:
1. Real-time drift detection for Triton Inference Server models.
2. Integration with Redis for pub/sub messaging and data storage.
3. Logging of drift predictions and statistics.
4. Support for multiple models and global drift tracking.

Main Components:
- Triton Client: Communicates with the Triton Inference Server for drift predictions.
- Redis Client: Handles pub/sub messaging and stores drift statistics.
- Drift Detection: Predicts drift using a dedicated model for each primary model.
- Drift Recording: Stores drift occurrences and updates counters in Redis.

Usage:
Run the script with the following command:
    python drift_detection.py [OPTIONS]

Options:
    --redis-host TEXT      Redis host (default: "localhost")
    --redis-port INTEGER   Redis port (default: 6379)
    --redis-db INTEGER     Redis database (default: 0)
    --triton-url TEXT      URL of the Triton Inference Server (default: "localhost:8000")

The script will then:
1. Connect to the specified Redis server and Triton Inference Server.
2. Subscribe to the 'inferences-updated' channel in Redis.
3. Process incoming messages, predict drift, and update statistics.
4. Log drift predictions and counts.

Redis Key Structure:
- Global drift counter: 'drifts:counter'
- Model-specific drift counter: 'drifts:{model_name}:counter'
- Global prediction counter: 'predictions:counter'
- Model-specific prediction counter: 'predictions:{model_name}:counter'
- Global drift occurrences: 'drifts:occurrences' (sorted set)
- Model-specific drift occurrences: 'drifts:{model_name}:occurrences' (sorted set)

Note: This script assumes that for each model '{model_name}' in the Triton server, 
there exists a corresponding drift detection model named '{model_name}-ddd'.

For more detailed information, refer to the function docstrings within the script.
"""

import typer
import redis
import json
import time
import logging
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from urllib.parse import urlparse
from datetime import datetime

app = typer.Typer()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def get_triton_client(server_url: str) -> httpclient.InferenceServerClient:
    """Initialize the Triton inference server client."""
    try:
        # Remove scheme if present
        parsed_url = urlparse(server_url)
        url_without_scheme = parsed_url.netloc or parsed_url.path
        return httpclient.InferenceServerClient(url=url_without_scheme)
    except Exception as e:
        logging.error(f"Client creation failed: {str(e)}")
        raise typer.Exit(code=1)

def predict_drift(client: httpclient.InferenceServerClient, model_name: str, input_data: np.ndarray) -> int:
    """Predict drift using the Triton Inference Server."""
    inputs = [httpclient.InferInput('float_input', input_data.shape, 'FP32')]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [httpclient.InferRequestedOutput('label')]

    try:
        result = client.infer(model_name, inputs, outputs=outputs)
        prediction = int(result.as_numpy('label')[0])
        if prediction not in [0, 1]:
            logging.warning(f"Unexpected drift prediction value: {prediction}. Expected 0 or 1.")
        return prediction
    except InferenceServerException as e:
        logging.error(f"Inference failed for model {model_name}: {str(e)}")
        return -1  # Indicate error
    except Exception as e:
        logging.error(f"Unexpected error during inference: {str(e)}")
        return -2  # Indicate unexpected error

def record_drift_occurrence(redis_client: redis.Redis, model_name: str, session_counter: int):
    """Record a drift occurrence with current timestamp for both global and model-specific occurrences."""
    try:
        current_time = int(time.time())
        pipeline = redis_client.pipeline()
        # Record global occurrence
        global_member = f"{model_name}:{session_counter}"
        pipeline.zadd("drifts:occurrences", {global_member: current_time})
        # Record model-specific occurrence
        pipeline.zadd(f"drifts:{model_name}:occurrences", {session_counter: current_time})
        pipeline.execute()
        logging.info(f"Recorded drift occurrence for {model_name}, session {session_counter} at {current_time}")
    except redis.RedisError as e:
        logging.error(f"Failed to record drift occurrence: {str(e)}")

def update_drift_counters(redis_client: redis.Redis, model_name: str, session_counter: int, drift_detected: bool):
    """Update global and per-model drift counters."""
    pipeline = redis_client.pipeline()
    try:
        if drift_detected:
            pipeline.incr('drifts:counter')
            pipeline.incr(f'drifts:{model_name}:counter')
            record_drift_occurrence(redis_client, model_name, session_counter)
            # Notify subscribers that drifts data has been updated
            key = f"drifts:{model_name}:{session_counter}"
            redis_client.publish('drifts-updated', f'{key}')
        pipeline.incr('predictions:counter')
        pipeline.incr(f'predictions:{model_name}:counter')
        pipeline.execute()
        
        global_drift_count = redis_client.get('drifts:counter') or 0
        model_drift_count = redis_client.get(f'drifts:{model_name}:counter') or 0
        global_total = redis_client.get('predictions:counter') or 0
        model_total = redis_client.get(f'predictions:{model_name}:counter') or 0
        
        global_drift_count = int(global_drift_count)
        model_drift_count = int(model_drift_count)
        global_total = int(global_total)
        model_total = int(model_total)
        
        logging.info(f"Global drift count: {global_drift_count}/{global_total}")
        logging.info(f"Model {model_name} drift count: {model_drift_count}/{model_total}")
    except (redis.RedisError, ValueError) as e:
        logging.error(f"Failed to update drift counters: {str(e)}")

def process_update(redis_client: redis.Redis, triton_client: httpclient.InferenceServerClient, key: str):
    """Process a single update from Redis."""
    start_time = datetime.now()
    logging.info(f"Processing update for key: {key}")
    
    parts = key.split(':')
    model_name, session_counter = parts[1], parts[2]

    # Get input data from Redis
    input_key = f"{key}:input_0"
    input_data_json = redis_client.get(input_key)
    if input_data_json is None:
        logging.warning(f"No input data found for key: {input_key}")
        return

    try:
        input_data = np.array(json.loads(input_data_json), dtype=np.float32)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON data for key {input_key}: {str(e)}")
        return
    except Exception as e:
        logging.error(f"Unexpected error while processing input data for key {input_key}: {str(e)}")
        return

    # Predict drift
    drift_prediction = predict_drift(triton_client, f"{model_name}-ddd", input_data)

    # Store drift prediction in Redis
    drift_key = f"drifts:{model_name}:{session_counter}"
    try:
        redis_client.set(drift_key, str(drift_prediction))
        logging.info(f"Stored drift prediction {drift_prediction} for key: {drift_key}")
        
        # Update drift counters
        update_drift_counters(redis_client, model_name, int(session_counter), drift_prediction == 1)
    except redis.RedisError as e:
        logging.error(f"Failed to store drift prediction in Redis: {str(e)}")
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logging.info(f"Processing completed in {processing_time:.2f} seconds")

@app.command()
def main(
    redis_host: str = typer.Option("localhost", help="Redis host"),
    redis_port: int = typer.Option(6379, help="Redis port"),
    redis_db: int = typer.Option(0, help="Redis database"),
    triton_url: str = typer.Option("localhost:8000", help="URL of the Triton Inference Server"),
):
    """
    Subscribe to Redis updates and perform drift detection using Triton Inference Server.
    """
    logging.info(f"Starting drift detection script with Redis: {redis_host}:{redis_port}, Triton: {triton_url}")
    
    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        triton_client = get_triton_client(triton_url)
    except Exception as e:
        logging.error(f"Failed to initialize clients: {str(e)}")
        raise typer.Exit(code=1)

    pubsub = redis_client.pubsub()
    pubsub.subscribe('inferences-updated')

    logging.info("Listening for updates...")

    try:
        for message in pubsub.listen():
            if message['type'] == 'message':
                key = message['data'].decode('utf-8')
                process_update(redis_client, triton_client, key)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        pubsub.unsubscribe()
        redis_client.close()
        logging.info("Drift detection script stopped")

if __name__ == "__main__":
    app()
