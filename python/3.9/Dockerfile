# docker build -t activeeon/python:3.9 .
# docker run -it --rm activeeon/python:3.9 bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/python:3.9 /bin/bash
FROM python:3.9

WORKDIR /tmp

RUN apt-get clean && apt-get update -y -qq

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
RUN apt-get install -y openjdk-11-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN python -m pip install --upgrade pip
RUN python -m pip install cython py4j pyyaml wget lxml validators

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
