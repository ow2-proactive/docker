# docker build -t activeeon/pytorch:cpu_latest .
# docker build -t activeeon/pytorch:cpu_1.11.0 .
# docker run -it --rm activeeon/pytorch bash
# docker run -it --rm -v $PWD:/home -w /home --network=host --shm-size=256M activeeon/pytorch /bin/bash
FROM activeeon/python:3.8

WORKDIR /tmp

# Install python libraries
# ------------------------
# RUN pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --no-cache-dir
# RUN pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --no-cache-dir
# RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install thop==0.0.31-2005241907

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
