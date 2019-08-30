FROM python:3.6

WORKDIR /root

#RUN ln -sf $(which python3) /usr/local/bin/python3

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
        libgl1-mesa-glx \
        && apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install java
# ------------
RUN apt-get update && apt-get install -y openjdk-11-jdk
RUN apt-get install ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

# Install python libraries
# ------------------------
RUN pip install cython
RUN pip install tensorflow keras

RUN pip install scikit-learn
RUN pip install pandas pandas-ml pandasql
RUN pip install networkx nltk pymc3 statsmodels luminol tsfresh

#RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
#RUN pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision==0.2.2.post2
RUN pip install torchtext gensim implicit

RUN pip install scikit-image graphviz opencv-contrib-python-headless
RUN pip install bokeh
RUN pip install boto boto3 bz2file s3fs

RUN pip install cassandra-driver cx-Oracle elasticsearch mysql-connector-python pymongo pymssql PyMySQL pyodbc pyOpenSSL sqlalchemy SecretStorage

RUN pip install tabulate colorama future
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN pip install beautifulsoup4 tqdm dill
RUN pip install psycopg2 psycopg2-binary

RUN pip install pantab
RUN pip install tableauserverclient

RUN pip install tensorboardX

RUN pip install connexion[swagger-ui]

# azure-cognitiveservices-vision-face
RUN pip install azure-cognitiveservices-vision-computervision azure-cognitiveservices-vision-contentmoderator azure-cognitiveservices-vision-customvision azure-cognitiveservices-search-websearch azure-cognitiveservices-search-imagesearch azure-cognitiveservices-search-entitysearch azure-cognitiveservices-search-customsearch azure-cognitiveservices-search-visualsearch azure-cognitiveservices-search-videosearch azure-cognitiveservices-search-newssearch  azure-cognitiveservices-language-textanalytics azure-cognitiveservices-language-spellcheck azure-mgmt-cognitiveservices
RUN pip install azure-mgmt-machinelearningcompute azure-mgmt-batchai
RUN pip install azureml-sdk==1.0.48

RUN pip install cntk
RUN pip install mxnet
RUN pip install pyspark

RUN pip install xgboost catboost
RUN pip install auto-sklearn
RUN pip install tpot
RUN pip install git+https://github.com/AIworx-Labs/chocolate@master
RUN pip install ghalton

RUN pip install dask distributed
RUN pip install dlib

RUN pip install -U jupyter jupyterlab
RUN pip install py4j wget lxml validators

RUN pip install eli5
#RUN pip uninstall matplotlib
RUN pip install matplotlib==2.2.2
RUN pip install mpld3
RUN pip install shap
RUN pip install pdpbox

RUN pip install flask

RUN pip uninstall numpy -y
RUN rm -rf /usr/local/lib/python3.6/site-packages/numpy
RUN pip install numpy==1.16.4

#RUN pip install proactive proactive-jupyter-kernel --upgrade
#RUN python -m proactive-jupyter-kernel.install

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .
RUN rm -rf visdom

# Install ODPI-C libraries for Oracle xe-11g database
# ----------------------------------------------
WORKDIR /
COPY instantclient-basic-linux.x64-11.2.0.4.0.zip /
RUN mkdir -p /opt/oracle && unzip instantclient-basic-linux.x64-11.2.0.4.0.zip  &&  mv instantclient_11_2/ /opt/oracle/ && rm  instantclient-basic-linux.x64-11.2.0.4.0.zip 
RUN sh -c "echo /opt/oracle/instantclient_11_2 > /etc/ld.so.conf.d/oracle-instantclient.conf"
RUN ldconfig

RUN rm -rf ~/.cache/pip

# Install Tableau Extract API
#------------------------------------------------
RUN wget -O SDK.tar.gz "https://downloads.tableau.com/tssoftware/extractapi-py-linux-x86_64-2019-1-2.tar.gz" \
      && mkdir -p /sdk \
      && tar -xzf SDK.tar.gz -C /sdk --strip-components=1 \
      && rm SDK.tar.gz \
      && cd /sdk \
      && python setup.py install \
      && rm -rf /sdk
