# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# docker build -t activeeon/nvidia:pytorch .
# docker run -it --rm --gpus all --privileged activeeon/nvidia:pytorch nvidia-smi
FROM nvcr.io/nvidia/pytorch:21.03-py3

# Update Ubuntu
# -------------
RUN apt-get update

# Install tzdata package
# ----------------------
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install java
# ------------
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN pip install py4j
RUN pip install wget validators
RUN pip install lxml

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN cd /tmp && git clone https://github.com/facebookresearch/visdom.git
RUN cd /tmp/visdom && pip install -e . && easy_install .
# WARNING: The easy_install command is deprecated and will be removed in a future version.

# Install GPU monitoring package
# https://github.com/anderskm/gputil
# ----------------------------------
RUN pip install gputil

# Clean
# -----
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
