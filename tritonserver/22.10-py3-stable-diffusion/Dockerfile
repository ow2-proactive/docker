# docker build -t activeeon/tritonserver:22.10-py3-stable-diffusion .
FROM nvcr.io/nvidia/tritonserver:22.10-py3

RUN python3 -m pip install --upgrade pip

RUN pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install --upgrade diffusers==0.2.4 scipy==1.9.1 transformers==4.21.2
