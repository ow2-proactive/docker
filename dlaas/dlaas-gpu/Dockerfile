# docker build -t activeeon/dlaas:gpu-latest .
FROM activeeon/cuda2

COPY ../dlaas_pb2.py /model_as_service
COPY ../dlaas.proto /model_as_service
COPY ../utils.py /model_as_service
WORKDIR /model_as_service

RUN pip install -r ../requirements.txt

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list 
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

RUN apt-get tree

