# docker build -t activeeon/tensorflow:latest .
FROM tensorflow/tensorflow:latest

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

RUN pip install py4j
RUN pip install keras --no-deps
RUN pip install pyyaml
RUN pip install scipy

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
