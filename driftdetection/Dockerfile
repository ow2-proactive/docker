# docker build --no-cache -t activeeon/driftdetection:latest .
# docker run -it --rm activeeon/driftdetection bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/driftdetection /bin/bash
FROM activeeon/python:3.8

# Update
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt upgrade -y

# Install python libraries
# ------------------------
COPY requirements.operationaldrift.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.operationaldrift.txt

RUN python3 -m pip cache purge
RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
