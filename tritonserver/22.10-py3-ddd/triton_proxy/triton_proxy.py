#!/usr/bin/env python3

"""
Triton Inference Server Proxy

This script implements a Flask-based proxy server for the Triton Inference Server.
It intercepts requests meant for the Triton server, logs and processes them,
forwards them to the actual Triton server, and then processes and logs the responses.

Key features:
1. Intercepts and logs all incoming requests to the Triton Inference Server.
2. Parses and logs JSON and binary data from requests and responses.
3. Handles various HTTP methods (GET, POST, PUT, DELETE, OPTIONS).
4. Processes binary data for both inputs and outputs, supporting different data types (e.g., FP32, INT64).
5. Stores request and response data in Redis for logging and analysis.
6. Implements error handling and logging.
7. Uses Typer for command-line argument parsing and configuration.
8. Maintains session consistency with a unique counter for each request-response cycle.

Usage:
Run the script with the desired command-line arguments:
python triton_proxy.py [OPTIONS]

Options:
--host TEXT                 Host to run the proxy server on (default: 0.0.0.0)
--port INTEGER              Port to run the proxy server on (default: 9090)
--triton-url TEXT           URL of the Triton Inference Server (default: http://localhost:8000)
--debug / --no-debug        Enable debug mode (default: False)
--max-content-length INTEGER  Maximum content length in bytes (default: 52428800 (50MB))
--redis-host TEXT           Redis host (default: localhost)
--redis-port INTEGER        Redis port (default: 6379)
--redis-db INTEGER          Redis database (default: 0)

Example:
./triton_proxy.py --host 0.0.0.0 --port 9090 --triton-url http://localhost:8000 --debug

Note: This proxy is designed for debugging and monitoring purposes and may impact performance.
Use in production environments should be carefully considered.
"""

import typer
import requests
import json
import redis
import time
import numpy as np
from datetime import datetime
from flask import Flask, request, Response
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Global variables to store configuration
config = {
    "TRITON_SERVER_URL": "http://localhost:8000",
    "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50 MB
    "REDIS_HOST": "localhost",
    "REDIS_PORT": 6379,
    "REDIS_DB": 0,
}

# Initialize Redis client
redis_client = None

def init_redis():
    """
    Initialize the Redis client with the configuration from the global config.
    This function should be called before any Redis operations are performed.
    """
    global redis_client
    try:
        redis_client = redis.Redis(host=config['REDIS_HOST'], port=config['REDIS_PORT'], db=config['REDIS_DB'])
        logging.info(f"Redis client initialized successfully: {config['REDIS_HOST']}:{config['REDIS_PORT']}")
    except redis.RedisError as e:
        logging.error(f"Failed to initialize Redis client: {str(e)}")
        raise typer.Exit(code=1)

def get_next_counter(redis_client: redis.Redis, model_name: str) -> tuple:
    """
    Atomically increment and return the next counter value for a given model and global counter.
    
    Args:
        redis_client (redis.Redis): The Redis client instance.
        model_name (str): Name of the model for which to get the next counter.
    
    Returns:
        tuple: (model_counter, global_counter) The next counter value for the specified model and the global counter.
    """
    try:
        pipeline = redis_client.pipeline()
        model_counter_key = f"inferences:{model_name}:counter"
        global_counter_key = "inferences:counter"
        
        pipeline.incr(model_counter_key)
        pipeline.incr(global_counter_key)
        
        model_counter, global_counter = pipeline.execute()
        
        return model_counter, global_counter
    except redis.RedisError as e:
        logging.error(f"Failed to get next counter for model {model_name} and global counter: {str(e)}")
        return -1, -1

def store_session_timestamp(model_name: str, session_counter: int, timestamp: int, prefix: str):
    """
    Store the timestamp for a specific session in Redis.

    Args:
        model_name (str): The name of the model.
        session_counter (int): The counter value for the current session.
        timestamp (int): The timestamp in milliseconds.
        prefix (str): The prefix to be used in the Redis key.
    """
    try:
        key = f"inferences:{model_name}:{session_counter}:timestamp:{prefix}"
        redis_client.set(key, timestamp)
        logging.debug(f"Stored {prefix} timestamp for session {session_counter}")
    except redis.RedisError as e:
        logging.error(f"Failed to store timestamp for session {session_counter}: {str(e)}")

def store_in_redis(model_name: str, data_type: str, data: dict, session_counter: int):
    """
    Store data in Redis with a session-preserved counter.
    
    Args:
        model_name (str): Name of the model.
        data_type (str): Type of data being stored (e.g., 'request_json', 'input_0').
        data (dict): The data to be stored.
        session_counter (int): Counter value for the current session.
    """
    try:
        key = f"inferences:{model_name}:{session_counter}:{data_type}"
        redis_client.set(key, json.dumps(data))
        logging.debug(f"Stored {data_type} data for session {session_counter}")
    except redis.RedisError as e:
        logging.error(f"Failed to store {data_type} data for session {session_counter}: {str(e)}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to serialize {data_type} data for session {session_counter}: {str(e)}")

def parse_binary_data(binary_data: bytes, entries: list) -> list:
    """
    Parse binary data into an array of values based on the provided datatype.
    
    Args:
        binary_data (bytes): Raw binary data to parse.
        entries (list): List of entries containing size and datatype information.
    
    Returns:
        list: List of parsed numpy arrays.
    
    Raises:
        ValueError: If an unsupported datatype is encountered.
    """
    outputs = []
    current_position = 0
    for entry in entries:
        try:
            output_size = entry.get('parameters', {}).get('binary_data_size', 0)
            datatype = entry.get('datatype', 'FP32')
            shape = entry.get('shape', [])
            if datatype == 'FP32':
                dtype = np.float32
                bytes_per_element = 4
            elif datatype == 'INT64':
                dtype = np.int64
                bytes_per_element = 8
            else:
                raise ValueError(f"Unsupported datatype: {datatype}")
            
            num_elements = output_size // bytes_per_element
            output_array = np.frombuffer(
                binary_data[current_position:current_position + output_size],
                dtype=dtype
            )
            if shape:
                output_array = output_array.reshape(shape)
            outputs.append(output_array)
            current_position += output_size
        except Exception as e:
            logging.error(f"Error parsing binary data: {str(e)}")
            outputs.append(None)
    return outputs

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path: str):
    """
    Main proxy function that handles all incoming requests.
    
    This function intercepts the request, extracts the model name, generates a session counter,
    processes the request body, forwards the request to the Triton server, and then processes
    the response.
    
    Args:
        path (str): The path of the incoming request.
    
    Returns:
        Response: Proxied response from the Triton server.
    """
    start_time = datetime.now()
    logging.info(f"Intercepted Request: {request.method} {request.url}")
    
    # Extract model name from the URL
    path_parts = path.split('/')
    model_name = 'unknown_model'
    for i, part in enumerate(path_parts):
        if part == 'models' and i + 1 < len(path_parts):
            model_name = path_parts[i + 1]
            break
    logging.info(f"Extracted model name: {model_name}")
    
    # Generate a single counter for this session
    model_counter, global_counter = get_next_counter(redis_client, model_name)
    session_counter = model_counter
    
    # Store the session start timestamp
    start_timestamp = int(time.time() * 1000)
    store_session_timestamp(model_name, session_counter, start_timestamp, "start")
    
    # Read and process the request body
    body = request.get_data()
    process_request_body(body, model_name, session_counter)
    
    # Forward the request to Triton Server
    triton_url = f"{config['TRITON_SERVER_URL']}/{path}"
    headers = dict(request.headers)
    headers.pop('Host', None)
    headers['Accept-Encoding'] = 'gzip'
    
    try:
        # Forward the request
        resp = requests.request(
            method=request.method,
            url=triton_url,
            headers=headers,
            data=body,
            allow_redirects=False
        )
        
        # Process the response
        resp_body = resp.content
        resp.headers.pop('Content-Encoding', None)
        process_response_body(resp_body, model_name, session_counter)
        
        # Store the session end timestamp
        end_timestamp = int(time.time() * 1000)
        store_session_timestamp(model_name, session_counter, end_timestamp, "end")

        # Notify subscribers that inferences data has been updated
        key = f"inferences:{model_name}:{session_counter}"
        redis_client.publish('inferences-updated', f'{key}')
        
        # Create and return the response
        response = Response(
            resp_body,
            status=resp.status_code,
            headers=resp.headers.items(),
            direct_passthrough=True
        )
        response.headers['Content-Length'] = str(len(resp_body))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logging.info(f"Request processed in {processing_time:.2f} seconds")
        
        return response
    except requests.RequestException as e:
        logging.error(f"Error forwarding request to Triton server: {str(e)}")
        return Response("Error forwarding request", status=500)

def process_request_body(body: bytes, model_name: str, session_counter: int):
    """
    Process the request body, extracting and logging JSON and binary data.
    
    Args:
        body (bytes): Raw request body.
        model_name (str): Name of the model being queried.
        session_counter (int): Unique counter for this request-response session.
    """
    json_end_index = find_json_end(body)
    if json_end_index != -1:
        json_bytes = body[:json_end_index]
        try:
            json_string = json_bytes.decode('utf-8')
            request_json = json.loads(json_string)
            logging.info(f'Request JSON:\n{request_json}')
            store_in_redis(model_name, 'request_json', request_json, session_counter)
            
            # Process binary data for inputs
            if json_end_index < len(body):
                binary_data = body[json_end_index:]
                logging.info(f'Binary data length: {len(binary_data)}')
                inputs = parse_binary_data(binary_data, request_json.get('inputs', []))
                for idx, input_array in enumerate(inputs):
                    if input_array is not None:
                        input_shape = request_json['inputs'][idx]['shape']
                        reshaped_input = np.array(input_array).reshape(input_shape)
                        logging.info(f'INPUT{idx} Values:\n{reshaped_input}')
                        store_in_redis(model_name, f'input_{idx}', reshaped_input.tolist(), session_counter)
                    else:
                        logging.warning(f'Failed to process INPUT{idx}')
        except Exception as e:
            logging.error(f'Error processing JSON part of request body: {str(e)}')
    else:
        logging.warning('Could not find the end of the JSON object in request body.')

def process_response_body(body: bytes, model_name: str, session_counter: int):
    """
    Process the response body, extracting and logging JSON and binary data.
    
    Args:
        body (bytes): Raw response body.
        model_name (str): Name of the model that was queried.
        session_counter (int): Unique counter for this request-response session.
    """
    json_end_index = find_json_end(body)
    if json_end_index != -1:
        json_bytes = body[:json_end_index]
        try:
            json_string = json_bytes.decode('utf-8')
            json_response = json.loads(json_string)
            logging.info(f'Parsed JSON Response:\n{json_response}')
            store_in_redis(model_name, 'response_json', json_response, session_counter)
            
            # Handle binary data part, if exists
            if json_end_index < len(body):
                binary_data = body[json_end_index:]
                logging.info(f'Binary Data Length: {len(binary_data)}')
                outputs = parse_binary_data(binary_data, json_response.get('outputs', []))
                for idx, output_array in enumerate(outputs):
                    if output_array is not None:
                        logging.info(f'OUTPUT{idx} Values:\n{output_array}')
                        store_in_redis(model_name, f'output_{idx}', output_array.tolist(), session_counter)
                    else:
                        logging.warning(f'Failed to process OUTPUT{idx}')
        except Exception as e:
            logging.error(f'Error parsing JSON part of response: {str(e)}')
            logging.debug(f'Attempted JSON Parse: {json_bytes.decode("utf-8", errors="replace")}')
    else:
        logging.warning('Could not find the end of the JSON object in response.')

def find_json_end(body_bytes: bytes) -> int:
    """
    Find the end index of the JSON object in a byte string.
    
    Args:
        body_bytes (bytes): Byte string containing JSON data.
    
    Returns:
        int: Index of the end of the JSON object, or -1 if not found.
    """
    brace_count = 0
    for i, byte in enumerate(body_bytes):
        if byte == ord('{'):
            brace_count += 1
        elif byte == ord('}'):
            brace_count -= 1
            if brace_count == 0:
                return i + 1
    return -1

def run_app(host: str, port: int, debug: bool):
    """
    Run the Flask application with the specified host, port, and debug settings.
    
    Args:
        host (str): The host to run the server on.
        port (int): The port to run the server on.
        debug (bool): Whether to run the server in debug mode.
    """
    app.run(host=host, port=port, debug=debug)

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions."""
    logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return "Internal Server Error", 500

def main(
    host: str = typer.Option("0.0.0.0", help="Host to run the proxy server on"),
    port: int = typer.Option(9090, help="Port to run the proxy server on"),
    triton_url: str = typer.Option("http://localhost:8000", help="URL of the Triton Inference Server"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
    max_content_length: int = typer.Option(52428800, help="Maximum content length in bytes (default: 50MB)"),
    redis_host: str = typer.Option("localhost", help="Redis host"),
    redis_port: int = typer.Option(6379, help="Redis port"),
    redis_db: int = typer.Option(0, help="Redis database")
):
    """
    Main function to set up and run the Triton Proxy server.
    
    This function updates the global configuration, initializes the Redis client,
    and starts the Flask application.
    
    Args:
        host (str): Host to run the proxy server on.
        port (int): Port to run the proxy server on.
        triton_url (str): URL of the Triton Inference Server.
        debug (bool): Enable debug mode.
        max_content_length (int): Maximum content length in bytes.
        redis_host (str): Redis host.
        redis_port (int): Redis port.
        redis_db (int): Redis database.
    """
    # Update global configuration
    config["TRITON_SERVER_URL"] = triton_url
    config["MAX_CONTENT_LENGTH"] = max_content_length
    config["REDIS_HOST"] = redis_host
    config["REDIS_PORT"] = redis_port
    config["REDIS_DB"] = redis_db
    
    # Set Flask app configuration
    app.config['MAX_CONTENT_LENGTH'] = config["MAX_CONTENT_LENGTH"]
    
    logging.info(f"Starting Triton Proxy Server:")
    logging.info(f"Host: {host}")
    logging.info(f"Port: {port}")
    logging.info(f"Triton Server URL: {config['TRITON_SERVER_URL']}")
    logging.info(f"Debug Mode: {'Enabled' if debug else 'Disabled'}")
    logging.info(f"Max Content Length: {config['MAX_CONTENT_LENGTH']} bytes")
    logging.info(f"Redis Host: {config['REDIS_HOST']}")
    logging.info(f"Redis Port: {config['REDIS_PORT']}")
    logging.info(f"Redis DB: {config['REDIS_DB']}")
    
    # Initialize Redis client
    init_redis()
    
    # Run the Flask app
    run_app(host, port, debug)

if __name__ == "__main__":
    typer.run(main)
