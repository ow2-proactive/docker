FROM python:3.8.10
WORKDIR /morphemic_project
#ENV DEBIAN_FRONTEND noninteractive
RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -y --no-install-recommends apt-utils
RUN python -m pip install --upgrade pip
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
# Install ML libraries
# ------------------------
RUN pip install tensorflow keras
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install matplotlib
RUN pip install numpy
RUN pip install plotly
RUN pip install scipy
RUN pip install convertdate
RUN pip install lunarcalendar
RUN pip install holidays
RUN pip install tqdm
# Install prophet
# ------------------------
RUN pip install pystan==2.19.1.1
RUN pip install fbprophet
#Install gluonts
#-------------------------
#RUN pip install mxnet
#RUN pip install gluonts
#Install ActiveMQ integration libraries
#-------------------------
RUN pip install stomp.py
RUN pip install PyYAML
RUN pip install Timeloop
RUN pip install python-slugify
#Install DatasetMaker integration libraries
#-------------------------
RUN pip install filelock
RUN pip install influxDB
# Force the installation of the version 0.10.8.1 of py4j
# This is due to the double connection from docker issue with the version 0.10.9
# ------------------------
RUN python -m pip install py4j==0.10.8.1
RUN rm -rf ~/.cache/pip
#Clone the prophet forecaster code
#-------------------------
RUN git clone -b prophet-imen https://github.com/dianajlailaty/morphemic_integration.git
RUN chmod +x /morphemic_project/morphemic_integration/forecasting_prophet/main.py
ENTRYPOINT python /morphemic_project/morphemic_integration/forecasting_prophet/main.py
