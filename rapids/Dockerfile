# docker build -t activeeon/rapidsai:cuda10.0-runtime-ubuntu16.04 .
# docker build -t activeeon/rapidsai:latest .
FROM rapidsai/rapidsai:cuda10.0-runtime-ubuntu16.04

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

RUN pip install py4j
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
#RUN conda install -c menpo opencv3

#RUN pip install keras --no-deps
#RUN pip install pyyaml
#RUN pip install scipy

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
RUN rm -rf visdom

RUN conda clean --all -y

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
