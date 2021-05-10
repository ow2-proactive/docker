# docker build -t activeeon/nvidia:rapidsai .
# docker run -it --rm --gpus all --privileged activeeon/nvidia:rapidsai nvidia-smi
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.18-cuda11.0-runtime-ubuntu18.04

# Activate conda environment in docker
# https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker
ENV CONDA_DEFAULT_ENV=rapids

# Update conda environment and install pip
#RUN conda update -n base -c defaults conda
#RUN conda install -c anaconda pip

# Update linux
RUN apt-get clean && apt-get update -y -qq

# Install generic packages
# ------------------------
RUN apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        swig \
        unzip \
        nano \
        libtbb-dev \
        openmpi-bin

# Install java
# ------------
RUN conda install -c anaconda openjdk

# Install python libraries
# ------------------------

# Fix conda paths
#RUN conda list
#RUN mv /conda/bin/python /conda/bin/python_base
#RUN mv /conda/bin/pip /conda/bin/pip_base
#RUN mv /conda/bin/conda /conda/bin/conda_base
#RUN ln -s /opt/conda/envs/rapids/bin/python /conda/bin/python
RUN mv /conda/bin/pip /conda/bin/pip.prev
RUN ln -s /opt/conda/envs/rapids/bin/pip /conda/bin/pip
#RUN ln -s /opt/conda/condabin/conda /conda/bin/conda
#RUN conda list
#RUN pip list

RUN pip install py4j
RUN pip install wget validators

# https://pytorch.org/get-started/locally/
# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# https://www.tensorflow.org/install/gpu
RUN pip install tensorflow-gpu==2.4.1 Keras==2.4.3
RUN pip install tensorboardX==2.1

RUN pip install catboost==0.24.4
#RUN pip install xgboost
#RUN pip install cntk
#RUN pip install mxnet
#RUN pip install pyspark

RUN pip install TPOT==0.11.7
#pip install auto-sklearn
RUN pip install git+https://github.com/ow2-proactive/chocolate@master
RUN pip install ghalton==0.6.2
RUN pip install hyperopt==0.2.5

RUN pip install eli5==0.11.0
RUN pip install mpld3==0.5.2
RUN pip install shap==0.38.1
RUN pip install PDPbox==0.2.0

#RUN pip install pymc3
#RUN pip install networkx nltk statsmodels luminol tsfresh
#RUN pip install torchtext gensim implicit
#RUN pip install dask distributed

RUN pip install opencv-contrib-python-headless==4.5.1.48
#RUN pip install scikit-image graphviz
#RUN pip install dlib
#RUN pip install Pillow==6.1

#RUN pip install bokeh
RUN pip install boto boto3 bz2file s3fs
RUN pip install colorama==0.4.4
#RUN pip install tabulate future tqdm
RUN pip install beautifulsoup4==4.9.3 dill==0.3.3

#RUN pip install flask flask-cors
#RUN pip install Werkzeug==0.16.1
#RUN pip install connexion[swagger-ui]==2.7.0

#RUN pip install -U jupyter jupyterlab
RUN pip install proactive

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN cd /tmp && git clone https://github.com/facebookresearch/visdom.git
RUN cd /tmp/visdom && pip install -e . && easy_install .

# Install GPU monitoring package
# https://github.com/anderskm/gputil
# ----------------------------------
RUN pip install gputil

# Clean
# -----
RUN conda clean --all -y
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
