# docker build -t activeeon/python:3.11 .
# docker run -it --rm activeeon/python:3.11 bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/python:3.11 /bin/bash
FROM python:3.11

WORKDIR /tmp

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt upgrade -y

# Install generic packages
# ------------------------
RUN apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        wget \
        git \
        curl \
        unzip \
        nano

# Install java
# ------------
# RUN apt-get install -y openjdk-11-jdk
# RUN apt-get install ca-certificates-java && update-ca-certificates -f
# ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
# RUN export JAVA_HOME
RUN mkdir -p /etc/apt/keyrings
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc
RUN echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt update
RUN apt install -y temurin-8-jdk
ENV JAVA_HOME /usr/lib/jvm/temurin-8-jdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN python -m pip install --upgrade pip
RUN python -m pip install cython py4j pyyaml wget lxml validators

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
