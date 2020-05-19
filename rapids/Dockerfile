# docker build -t activeeon/rapidsai:latest .
#FROM rapidsai/rapidsai:cuda10.0-runtime-ubuntu18.04
FROM rapidsai/rapidsai:0.13-cuda10.1-base-ubuntu18.04-py3.7

RUN apt-get clean && apt-get update -y -qq

# Install generic packages
# ------------------------
RUN apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        wget \
        swig \
        git \
        curl \
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

RUN apt-get install openmpi-bin -y

# Install java
# ------------
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Install python libraries
# ------------------------
RUN pip install py4j
RUN pip install wget validators

# https://pytorch.org/get-started/locally/
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# https://www.tensorflow.org/install/gpu
RUN pip install tensorflow-gpu keras
RUN pip install tensorboardX

RUN pip install catboost
#RUN pip install xgboost
#RUN pip install cntk
#RUN pip install mxnet
#RUN pip install pyspark

RUN pip install tpot
#pip install auto-sklearn
RUN pip install git+https://github.com/ow2-proactive/chocolate@master
RUN pip install ghalton
RUN pip install hyperopt

RUN pip install eli5
RUN pip install mpld3
#RUN pip install shap
RUN pip install pdpbox

#RUN pip install pymc3
#RUN pip install networkx nltk statsmodels luminol tsfresh
#RUN pip install torchtext gensim implicit
#RUN pip install dask distributed

RUN pip install opencv-contrib-python-headless
#RUN pip install scikit-image graphviz
#RUN pip install dlib
#RUN pip install Pillow==6.1

#RUN pip install bokeh
RUN pip install boto boto3 bz2file s3fs
RUN pip install colorama
#RUN pip install tabulate future tqdm
RUN pip install beautifulsoup4 dill

RUN pip install flask flask-cors
#RUN pip install Werkzeug==0.16.1
RUN pip install connexion[swagger-ui]==2.7.0

#RUN pip install -U jupyter jupyterlab
RUN pip install proactive

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
RUN rm -rf visdom

RUN conda clean --all -y

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
