# docker build -t activeeon/cuda:latest .
#FROM nvidia/cuda:latest
#FROM nvidia/cuda:10.0-cudnn7-devel
FROM nvidia/cuda:10.0-cudnn7-runtime

WORKDIR /root

RUN apt-get update 

# Install python
RUN apt-get install -y python3 python3-pip python3-dev python3-setuptools

# Install generic packages
# ------------------------
RUN apt-get install -y \
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

#RUN apt-get autoclean && apt-get clean \
#        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install python
#RUN apt-get install python3 python3-pip python3-dev
RUN ln -s /usr/bin/python3 /usr/local/bin/python
RUN which python
RUN which python3
RUN python -V
RUN python3 -V

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN which pip
RUN which pip3
RUN pip -V
RUN pip3 -V
#RUN ln -s /usr/bin/pip3 /usr/local/bin/pip

# Install java
# ------------
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
#RUN apt-get clean

# Install python libraries
# ------------------------
RUN pip install cython
RUN pip install py4j

RUN pip install tensorflow-gpu
RUN pip install pyyaml
RUN pip install scipy
RUN pip install keras --no-deps

RUN pip install scikit-learn
RUN pip install pandas pandas-ml pandasql
RUN pip install networkx nltk pymc3 statsmodels luminol tsfresh

RUN pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
#RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install torchtext gensim implicit

RUN pip install eli5 shap
RUN pip install scikit-image graphviz opencv-contrib-python-headless
RUN pip install bokeh
RUN pip install boto boto3 bz2file s3fs

#RUN pip install cassandra-driver cx-Oracle elasticsearch mysql-connector-python pymongo pymssql PyMySQL pyodbc pyOpenSSL sqlalchemy SecretStorage

RUN pip install tabulate colorama future
#RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN pip install beautifulsoup4 tqdm dill
#RUN pip install psycopg2 psycopg2-binary

# azure-cognitiveservices-vision-face
#RUN pip install azure-cognitiveservices-vision-computervision azure-cognitiveservices-vision-contentmoderator azure-cognitiveservices-vision-customvision azure-cognitiveservices-search-websearch azure-cognitiveservices-search-imagesearch azure-cognitiveservices-search-entitysearch azure-cognitiveservices-search-customsearch azure-cognitiveservices-search-visualsearch azure-cognitiveservices-search-videosearch azure-cognitiveservices-search-newssearch  azure-cognitiveservices-language-textanalytics azure-cognitiveservices-language-spellcheck azure-mgmt-cognitiveservices
#RUN pip install azure-mgmt-machinelearningcompute azure-mgmt-batchai

# https://github.com/Microsoft/CNTK/issues/3499
#RUN pip install cntk-gpu
RUN apt-get install openmpi-bin -y
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.20.10.1 /usr/lib/x86_64-linux-gnu/libmpi.so.12
RUN pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.6-cp36-cp36m-linux_x86_64.whl

RUN pip install mxnet
RUN pip install pyspark

RUN pip install xgboost catboost
RUN pip install auto-sklearn
RUN pip install tpot
RUN pip install git+https://github.com/AIworx-Labs/chocolate@master
#RUN pip install ghalton
RUN pip install cudf-cuda100==0.5 cuml-cuda100==0.5

RUN pip install dask distributed
RUN pip install dlib

RUN pip install -U jupyter jupyterlab
RUN pip install wget lxml validators
RUN pip install proactive

RUN pip install tensorboardX

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
#RUN pip install setuptools
#RUN which easy_install3
RUN git clone https://github.com/facebookresearch/visdom.git
#RUN cd visdom && pip install -e . && easy_install .
RUN cd visdom && pip install -e .
#RUN rm -rf visdom

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
