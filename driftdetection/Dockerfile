# docker build --no-cache -t activeeon/driftdetection:latest .
# docker run -it --rm activeeon/driftdetection bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/driftdetection /bin/bash
FROM activeeon/python:3.8

# Update
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt upgrade -y

# Install python libraries
# ------------------------
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy==1.20.3
RUN python3 -m pip install scikit-learn==1.3.2
RUN python3 -m pip install scikit-multiflow==0.5.3
RUN python3 -m pip install prometheus-api-client==0.5.4
RUN python3 -m pip install requests==2.31.0

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
