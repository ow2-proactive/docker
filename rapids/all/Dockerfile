# docker build -t activeeon/rapidsai:latest .
FROM rapidsai/rapidsai:cuda10.0-runtime-ubuntu18.04

# Install generic packages
# ------------------------
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        wget \
        swig \
        git \
        unzip \
        libaio1 \
        nano \
        freetds-dev \
        unixodbc \
        unixodbc-dev \
        libjpeg-dev \
        libtiff5-dev \
        libpng-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libtbb2 \
        libtbb-dev \
        libgl1-mesa-glx

# Install java
# ------------
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN pip install py4j
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN pip install torchtext gensim implicit

RUN pip install networkx nltk pymc3 statsmodels luminol tsfresh
RUN pip install tabulate colorama future
RUN pip install beautifulsoup4 tqdm dill
RUN pip install xgboost catboost

RUN pip install auto-sklearn
RUN pip install tpot
RUN pip install git+https://github.com/AIworx-Labs/chocolate@master

RUN pip install wget lxml validators
RUN pip install tensorboardX

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
RUN rm -rf visdom

RUN conda clean --all -y

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
