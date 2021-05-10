# https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
# docker build -t activeeon/nvidia:tensorflow .
# docker run -it --rm --gpus all --privileged activeeon/nvidia:tensorflow nvidia-smi
FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN pip install py4j
RUN pip install wget validators lxml
RUN pip install keras
RUN pip install scikit-learn

# Install GPU monitoring package
# https://github.com/anderskm/gputil
# ----------------------------------
RUN pip install gputil

# Clean
# -----
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
