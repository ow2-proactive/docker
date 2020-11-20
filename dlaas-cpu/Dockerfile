# docker build -t imenbizid/dlaas-cpu .
FROM activeeon/dlm3

COPY . /model_as_service
WORKDIR /model_as_service

RUN pip install -r requirements.txt

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list 
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

#RUN git clone -b r2.3 https://github.com/tensorflow/serving.git
#RUN cd serving
#RUN bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

#RUN /dl_model_as_service/tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...
#RUN /dl_model_as_service/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

