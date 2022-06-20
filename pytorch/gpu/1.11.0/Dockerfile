# docker build -t activeeon/pytorch:gpu_latest .
# docker build -t activeeon/pytorch:gpu_1.11.0 .
# docker run -it --rm activeeon/pytorch:gpu_latest bash
# docker run -it --rm -v $PWD:/home -w /home --network=host --shm-size=256M activeeon/pytorch:gpu_latest /bin/bash
#FROM activeeon/python:3.8
FROM activeeon/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /tmp

# Install python libraries
# ------------------------
# RUN pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --no-cache-dir
# RUN pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --no-cache-dir
# RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
# RUN pip3 install torch torchvision torchaudio
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install thop==0.0.31-2005241907
#RUN pip3 install thop==0.1.0.post2206102148
#RUN pip3 install onnx==1.12.0

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
