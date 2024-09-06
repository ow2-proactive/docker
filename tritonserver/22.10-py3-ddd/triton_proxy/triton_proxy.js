const express = require('express');
const httpProxy = require('http-proxy');
const bodyParser = require('body-parser');
const app = express();
const proxy = httpProxy.createProxyServer({});
const TRITON_SERVER_URL = 'http://localhost:8000'; // Change this to your Triton server URL

app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.raw({type: 'application/octet-stream', limit: '50mb'}));

// Middleware to log requests and input values
app.use((req, res, next) => {
    console.log('Intercepted Request:', req.method, req.url);
    let bodyChunks = [];
    req.on('data', (chunk) => {
        bodyChunks.push(chunk);
    }).on('end', () => {
        const bodyBuffer = Buffer.concat(bodyChunks);
        // Attempt to find the end of the JSON object
        let braceCount = 0;
        let jsonEndIndex = -1;
        for (let i = 0; i < bodyBuffer.length; i++) {
            if (bodyBuffer[i] === 123) { // '{'.charCodeAt(0)
                braceCount++;
            } else if (bodyBuffer[i] === 125) { // '}'.charCodeAt(0)
                braceCount--;
                if (braceCount === 0) {
                    jsonEndIndex = i + 1;
                    break;
                }
            }
        }
        if (jsonEndIndex !== -1) {
            const jsonString = bodyBuffer.slice(0, jsonEndIndex).toString();
            try {
                const requestJson = JSON.parse(jsonString);
                console.log('Request JSON:', requestJson);
                // Process binary data for inputs
                if (jsonEndIndex < bodyBuffer.length) {
                    const binaryData = bodyBuffer.slice(jsonEndIndex);
                    console.log('Binary data length:', binaryData.length);
                    const inputs = requestJson.inputs.map(() => []);
                    let currentPosition = 0;
                    requestJson.inputs.forEach((input, index) => {
                        const inputSize = input.parameters.binary_data_size;
                        for (let i = currentPosition; i < currentPosition + inputSize; i += 4) {
                            const floatValue = binaryData.readFloatLE(i);
                            inputs[index].push(floatValue);
                        }
                        currentPosition += inputSize; // Move to the next input's position
                    });
                    inputs.forEach((inputArray, index) => {
                        console.log(`INPUT${index} Values:`, inputArray);
                    });
                }
            } catch (error) {
                console.error('Error processing JSON part of request body:', error);
            }
        } else {
            console.error('Could not find the end of the JSON object.');
        }
    });
    next();
});

// Forward all requests to Triton Inference Server and intercept responses
app.use((req, res) => {
    proxy.web(req, res, { target: TRITON_SERVER_URL });
});

// Listen for the `proxyRes` event on the proxy to log responses.
proxy.on('proxyRes', function(proxyRes, req, res) {
    let originalBody = Buffer.from([]);
    proxyRes.on('data', function(data) {
        originalBody = Buffer.concat([originalBody, data]);
    });
    proxyRes.on('end', function() {
        // Accurately find the end of the JSON part by matching braces
        let braceCount = 0;
        let jsonEndIndex = 0;
        for (let i = 0; i < originalBody.length; i++) {
            if (originalBody[i] === 123) { // '{'.charCodeAt(0)
                braceCount++;
            } else if (originalBody[i] === 125) { // '}'.charCodeAt(0)
                braceCount--;
                if (braceCount === 0) {
                    jsonEndIndex = i + 1;
                    break;
                }
            }
        }
        const jsonString = originalBody.slice(0, jsonEndIndex).toString();
        try {
            // Parse the JSON part of the response
            const jsonResponse = JSON.parse(jsonString);
            console.log('Parsed JSON Response:', jsonResponse);
            // Handle binary data part, if exists
            if (jsonEndIndex < originalBody.length) {
                const binaryData = originalBody.slice(jsonEndIndex);
                console.log('Binary Data Length:', binaryData.length);
                // Assuming the binary data contains 32-bit floats
                // Initialize an empty array for each output
                const outputs = jsonResponse.outputs.map(() => []);
                // Track the current position in the binary data
                let currentPosition = 0;
                // Iterate through each output in the JSON response
                jsonResponse.outputs.forEach((output, index) => {
                    // Assuming each output is a flat array of 32-bit floats
                    const outputSize = output.parameters.binary_data_size;
                    for (let i = currentPosition; i < currentPosition + outputSize; i += 4) {
                        const floatValue = binaryData.readFloatLE(i);
                        outputs[index].push(floatValue);
                    }
                    currentPosition += outputSize; // Move to the next output's position
                });
                // Log each output separately
                outputs.forEach((outputArray, index) => {
                    console.log(`OUTPUT${index} Values:`, outputArray);
                });
            }
        } catch (error) {
            // Error parsing the JSON part
            console.error('Error parsing JSON part of response:', error);
            console.log('Attempted JSON Parse:', jsonString);
        }
    });
});

const PORT = 8080; // The port your proxy server will listen on
app.listen(PORT, () => {
    console.log(`Proxy server listening on port ${PORT}`);
});
