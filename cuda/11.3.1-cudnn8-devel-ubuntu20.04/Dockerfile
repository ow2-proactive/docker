# docker build -t activeeon/cuda:11.3.1-cudnn8-devel-ubuntu20.04 .
# docker run -it --rm activeeon/cuda:11.3.1-cudnn8-devel-ubuntu20.04 bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/cuda:11.3.1-cudnn8-devel-ubuntu20.04 /bin/bash
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /tmp

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get clean && apt-get update -y -qq

# Install generic packages
# ------------------------
RUN apt-get install -y \
	python3 \
	python3-dev \
	python3-pip \
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
RUN pip install cython py4j pyyaml wget lxml validators

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
