# docker build --no-cache -t activeeon/dlm4:latest .
# docker run -it --rm activeeon/dlm4 bash
#
# Package    Version
# ---------- --------
# Cython     0.29.26
# decorator  5.1.0
# lxml       4.7.1
# pip        21.2.4
# py4j       0.10.9.3
# PyYAML     6.0
# setuptools 57.5.0
# six        1.16.0
# validators 0.18.2
# wget       3.2
# wheel      0.37.0
#
FROM activeeon/python:3.8

# Update
RUN apt-get clean && apt-get update -y -qq

# Install python libraries
# ------------------------
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install requests==2.31.0
RUN python3 -m pip install numpy==1.24.3
RUN python3 -m pip install scipy==1.10.1
RUN python3 -m pip install scikit-learn==1.0.2
RUN python3 -m pip install onnxruntime==1.15.1
RUN python3 -m pip install protobuf==3.20.2
RUN python3 -m pip install skl2onnx==1.14.1
RUN python3 -m pip install catboost==1.2
RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install transformers==4.31.0

# https://github.com/triton-inference-server/client
# RUN python3 -m pip install tritonclient
# RUN python3 -m pip install "tritonclient[all]"
# RUN python3 -m pip install "tritonclient[http]"
# RUN python3 -m pip install "tritonclient[grpc]"
RUN python3 -m pip install "tritonclient[all]==2.36.0"

RUN python3 -m pip freeze > requirements.txt

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
