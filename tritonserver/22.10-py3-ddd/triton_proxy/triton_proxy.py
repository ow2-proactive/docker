import requests
import json
import struct

from flask import Flask, request, Response

app = Flask(__name__)

TRITON_SERVER_URL = 'http://localhost:8000'  # Change this to your Triton server URL

# Set maximum content length (e.g., 50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Define a route that matches all paths
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    # Intercepted request
    print('Intercepted Request:', request.method, request.url)

    # Read the request body
    body = request.get_data()

    # Process the body to parse the JSON part and binary data
    body_bytes = body

    # Attempt to find the end of the JSON object
    brace_count = 0
    json_end_index = -1

    for i in range(len(body_bytes)):
        if body_bytes[i] == ord('{'):
            brace_count += 1
        elif body_bytes[i] == ord('}'):
            brace_count -= 1
            if brace_count == 0:
                json_end_index = i + 1
                break

    if json_end_index != -1:
        json_bytes = body_bytes[:json_end_index]
        try:
            json_string = json_bytes.decode('utf-8')
            request_json = json.loads(json_string)
            print('Request JSON:', request_json)
            # Process binary data for inputs
            if json_end_index < len(body_bytes):
                binary_data = body_bytes[json_end_index:]
                print('Binary data length:', len(binary_data))
                inputs = []
                current_position = 0
                for input_entry in request_json.get('inputs', []):
                    input_size = input_entry.get('parameters', {}).get('binary_data_size', 0)
                    input_array = []
                    for i in range(current_position, current_position + input_size, 4):
                        float_bytes = binary_data[i:i+4]
                        if len(float_bytes) == 4:
                            float_value = struct.unpack('<f', float_bytes)[0]
                            input_array.append(float_value)
                    inputs.append(input_array)
                    current_position += input_size
                for idx, input_array in enumerate(inputs):
                    print(f'INPUT{idx} Values:', input_array)
        except Exception as e:
            print('Error processing JSON part of request body:', e)
    else:
        print('Could not find the end of the JSON object.')

    # Forward the request to Triton Server
    triton_url = f"{TRITON_SERVER_URL}/{path}"
    headers = dict(request.headers)
    # Remove 'Host' header if present, requests will set it correctly
    headers.pop('Host', None)
    resp = requests.request(
        method=request.method,
        url=triton_url,
        headers=headers,
        data=body,
        allow_redirects=False,
        stream=True
    )

    # Intercept the response
    resp_body = resp.content

    # Process the response body
    body_bytes = resp_body

    # Accurately find the end of the JSON part by matching braces
    brace_count = 0
    json_end_index = -1

    for i in range(len(body_bytes)):
        if body_bytes[i] == ord('{'):
            brace_count += 1
        elif body_bytes[i] == ord('}'):
            brace_count -= 1
            if brace_count == 0:
                json_end_index = i + 1
                break

    if json_end_index != -1:
        json_bytes = body_bytes[:json_end_index]
        try:
            json_string = json_bytes.decode('utf-8')
            json_response = json.loads(json_string)
            print('Parsed JSON Response:', json_response)
            # Handle binary data part, if exists
            if json_end_index < len(body_bytes):
                binary_data = body_bytes[json_end_index:]
                print('Binary Data Length:', len(binary_data))
                outputs = []
                current_position = 0
                for output_entry in json_response.get('outputs', []):
                    output_size = output_entry.get('parameters', {}).get('binary_data_size', 0)
                    output_array = []
                    for i in range(current_position, current_position + output_size, 4):
                        float_bytes = binary_data[i:i+4]
                        if len(float_bytes) == 4:
                            float_value = struct.unpack('<f', float_bytes)[0]
                            output_array.append(float_value)
                    outputs.append(output_array)
                    current_position += output_size
                for idx, output_array in enumerate(outputs):
                    print(f'OUTPUT{idx} Values:', output_array)
        except Exception as e:
            print('Error parsing JSON part of response:', e)
            print('Attempted JSON Parse:', json_bytes.decode('utf-8', errors='replace'))
    else:
        print('Could not find the end of the JSON object in response.')

    # Create a response to send back to the client
    response = Response(resp.content, status=resp.status_code)
    # Copy headers from Triton response
    for key, value in resp.headers.items():
        response.headers[key] = value
    return response

if __name__ == '__main__':
    PORT = 9090  # The port your proxy server will listen on
    app.run(host='0.0.0.0', port=PORT)
