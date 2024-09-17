#!/usr/bin/env python3

import sys
import typer
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import *

app = typer.Typer()

@app.command()
def main(
    server_url: str,
    model_name: str = "sum",
    shape: list[int] = typer.Option([1], help="Shape of the input tensors")
):
    
    with httpclient.InferenceServerClient(server_url) as client:
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)
        inputs = [
            httpclient.InferInput("INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)),
            httpclient.InferInput("INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype))
        ]
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0")
        ]
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")
        print(
            "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(input0_data, input1_data, output0_data)
        )
        if not np.allclose(input0_data + input1_data, output0_data):
            print("add_sub example error: incorrect sum")
            sys.exit(1)
        print("PASS: sum")
        sys.exit(0)

if __name__ == "__main__":
    typer.run(main)
