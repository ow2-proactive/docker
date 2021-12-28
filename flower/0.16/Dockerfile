# docker build -t activeeon/flower:latest .
# docker build -t activeeon/flower:0.16 .
# docker run -it --rm activeeon/flower bash
# docker run -it --rm -v $PWD:/home -w /home --network=host --shm-size=256M activeeon/flower /bin/bash
FROM activeeon/python:3.8

WORKDIR /tmp

# Install python libraries
# ------------------------
RUN pip install flwr==0.16.0 --no-cache-dir
# RUN pip install flwr==0.17.0 --no-cache-dir

RUN pip install openml==0.12.2 --no-cache-dir

RUN pip install tensorflow==2.5.0 --no-cache-dir
# RUN pip install tensorflow==2.6.2 --no-cache-dir

RUN pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 --no-cache-dir
# RUN pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --no-cache-dir
# RUN pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*