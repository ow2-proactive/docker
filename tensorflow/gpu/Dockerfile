# docker build -t activeeon/tensorflow:latest-gpu .
FROM tensorflow/tensorflow:latest-gpu

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

# Install python libraries
# ------------------------
RUN pip install py4j
RUN pip install keras
RUN pip install pyyaml
RUN pip install scipy

# Install GPU monitoring package
# https://github.com/anderskm/gputil
# ----------------------------------
RUN pip install gputil

# Clean
# -----
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
