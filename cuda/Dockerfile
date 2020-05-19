# docker build -t activeeon/cuda:latest .
# docker run -it --rm --gpus all --privileged activeeon/cuda nvidia-smi
FROM nvidia/cuda:10.0-cudnn7-runtime

WORKDIR /tmp

RUN apt-get clean && apt-get update -y -qq

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

# Install python
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py

# Install python libraries
# ------------------------
RUN pip install setuptools --upgrade
RUN pip install cython py4j pyyaml wget lxml validators

# https://www.tensorflow.org/install/gpu
RUN pip install tensorflow-gpu==1.15.2 keras

# https://pytorch.org/get-started/locally/
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
#RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision==0.2.2.post2
RUN pip install tensorboardX

# https://rapids.ai/start.html#get-rapids
RUN pip install cudf-cuda100==0.5 cuml-cuda100==0.5
RUN pip install dask distributed

RUN pip install scikit-learn
#RUN pip install scikit-learn==0.21  # to be compatible with auto-sklearn
RUN pip install pandas
#RUN pip install pandas-ml pandasql
#RUN pip uninstall -y enum34  # pandas-ml

RUN pip install xgboost catboost
#RUN pip install cntk
#RUN pip install mxnet
#RUN pip install pyspark

RUN pip install tpot
#RUN pip install tpot==0.11.0  # to be compatible with scikit-learn==0.21
#RUN pip install auto-sklearn  # not working
RUN pip install git+https://github.com/ow2-proactive/chocolate@master
RUN pip install ghalton
RUN pip install hyperopt

#RUN pip install pymc3
RUN pip install networkx torchtext nltk gensim implicit statsmodels luminol tsfresh

RUN pip install eli5
#RUN pip uninstall matplotlib
RUN pip install matplotlib==2.2.2
RUN pip install mpld3
RUN pip install shap==0.30.0
RUN pip install pdpbox

RUN pip install scikit-image graphviz opencv-contrib-python-headless
#RUN pip install dlib
#RUN pip install Pillow==6.1

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
#RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1
#RUN ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.20.10.1 /usr/lib/x86_64-linux-gnu/libmpi.so.12
#RUN pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.6-cp36-cp36m-linux_x86_64.whl

RUN pip install flask flask-cors
RUN pip install Werkzeug==0.16.1
RUN pip install connexion[swagger-ui]==2.6.0

#RUN pip uninstall numpy -y
#RUN rm -rf /usr/local/lib/python3.6/site-packages/numpy
#RUN pip install numpy==1.16.4

#RUN pip install -U jupyter jupyterlab
RUN pip install proactive

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
#WARNING: The easy_install command is deprecated and will be removed in a future version.

RUN conda clean --all -y

RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
