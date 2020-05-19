# docker build -t activeeon/cuda2 .
# docker run -it --rm --gpus all --privileged activeeon/cuda2 nvidia-smi
FROM nvidia/cuda:10.1-cudnn7-runtime

WORKDIR /tmp

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
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# https://repo.anaconda.com/archive/
ENV LATEST_CONDA="2020.02"
ENV PATH="/opt/anaconda3/bin:${PATH}"
RUN export PATH
RUN curl --silent -O https://repo.anaconda.com/archive/Anaconda3-$LATEST_CONDA-Linux-x86_64.sh \
    && bash Anaconda3-$LATEST_CONDA-Linux-x86_64.sh -b -p /opt/anaconda3
RUN conda update -n base -c defaults conda
#RUN conda install mamba -c conda-forge

# https://docs.conda.io/en/latest/miniconda.html#linux-installers
# https://hub.docker.com/r/continuumio/miniconda3/dockerfile
#ENV PATH="/opt/miniconda3/bin:${PATH}"
#RUN export PATH
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#    /bin/bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
#    rm /tmp/miniconda.sh && \
#    /opt/miniconda3/bin/conda clean -tipsy && \
#    ln -s /opt/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc
#RUN curl --silent -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh \
#    && bash Miniconda3-latest-Linux-x86.sh -b -p /opt/miniconda3

# Install python libraries
# ------------------------
# https://rapids.ai/start.html#get-rapids
#RUN conda install -c rapidsai -c nvidia -c conda-forge \
#    -c defaults rapids=0.13 python=3.7 cudatoolkit=10.2
#RUN conda install -c rapidsai-nightly -c nvidia -c conda-forge \
#    -c defaults rapids=0.14 python=3.7 cudatoolkit=10.2

# https://pytorch.org/get-started/locally/
# RUN pip install torch torchvision
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN pip install py4j
RUN pip install wget validators

# https://www.tensorflow.org/install/gpu
RUN pip install tensorflow-gpu keras
RUN pip install tensorboardX
RUN pip install xgboost catboost
#RUN pip install cntk
#RUN pip install mxnet
#RUN pip install pyspark

RUN pip install tpot
RUN pip install auto-sklearn
RUN pip install git+https://github.com/ow2-proactive/chocolate@master
RUN pip install ghalton
RUN pip install hyperopt

RUN pip install eli5
#RUN pip uninstall matplotlib
#RUN pip install matplotlib==2.2.2
RUN pip install mpld3
#RUN pip install shap==0.30.0
RUN pip install shap
RUN pip install pdpbox

#RUN pip install pymc3
RUN pip install networkx nltk statsmodels luminol tsfresh
RUN pip install torchtext gensim implicit
#RUN pip install dask distributed

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
