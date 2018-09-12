FROM python:3.6

WORKDIR /root

# Install generic packages
# ------------------------
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        wget \
        git \
        unzip \
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
        libgl1-mesa-glx \
        && apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-8-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

# Install python libraries
# ------------------------
RUN pip install cython
RUN pip install tensorflow keras

RUN pip install scikit-learn
RUN pip install pandas pandas-ml pandasql
RUN pip install networkx nltk pymc3 statsmodels luminol tsfresh

RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install torchtext gensim implicit

RUN pip install scikit-image graphviz opencv-contrib-python-headless
RUN pip install bokeh
RUN pip install boto boto3 bz2file s3fs

RUN pip install cassandra-driver cx-Oracle elasticsearch mysql-connector-python pymongo pymssql PyMySQL pyodbc pyOpenSSL sqlalchemy SecretStorage

RUN pip install tabulate colorama future
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN pip install beautifulsoup4 tqdm dill
RUN pip install psycopg2 psycopg2-binary

# azure-cognitiveservices-vision-face
RUN pip install azure-cognitiveservices-vision-computervision azure-cognitiveservices-vision-contentmoderator azure-cognitiveservices-vision-customvision azure-cognitiveservices-search-websearch azure-cognitiveservices-search-imagesearch azure-cognitiveservices-search-entitysearch azure-cognitiveservices-search-customsearch azure-cognitiveservices-search-visualsearch azure-cognitiveservices-search-videosearch azure-cognitiveservices-search-newssearch  azure-cognitiveservices-language-textanalytics azure-cognitiveservices-language-spellcheck azure-mgmt-cognitiveservices

RUN pip install azure-mgmt-machinelearningcompute azure-mgmt-batchai

RUN pip install cntk
RUN pip install mxnet
RUN pip install pyspark

RUN pip install dask distributed
RUN pip install dlib

RUN pip install -U jupyter jupyterlab
RUN pip install py4j wget lxml validators
RUN pip install proactive

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
RUN rm -rf visdom

RUN rm -rf ~/.cache/pip
