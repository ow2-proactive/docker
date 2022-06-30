# docker build -t activeeon/maas_dl:latest .
FROM activeeon/dlm3

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=HEAD

COPY dlaas_pb2.py /model_as_service/dlaas_pb2.py
COPY dlaas.proto /model_as_service/dlaas.proto
COPY utils.py /model_as_service/utils.py
COPY templates/ /model_as_service/templates/
COPY static/ /model_as_service/static/
COPY assets/ /model_as_service/assets/
COPY dash_utils.py /model_as_service/dash_utils.py
COPY requirements.txt /model_as_service/requirements.txt
WORKDIR /model_as_service

RUN pip install -r requirements.txt

# Installing Tensorflow serving not from sources
# RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list 
# RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
# RUN apt-get update && apt-get install tensorflow-model-server

RUN pip3 --no-cache-dir install \
    future>=0.17.1 \
    grpcio \
    h5py \
    keras_applications>=1.0.8 \
    keras_preprocessing>=1.1.0 \
    mock \
    numpy \
    portpicker \
    requests \
    --ignore-installed setuptools \
    --ignore-installed six>=1.12.0

# Set up Bazel
ENV BAZEL_VERSION 5.1.1
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download TF Serving sources (optionally at specific commit).
WORKDIR /tensorflow-serving
RUN curl -sSL --retry 5 https://github.com/tensorflow/serving/tarball/${TF_SERVING_VERSION_GIT_COMMIT} | tar --strip-components=1 -xzf -

#FROM base_build as binary_build
# Build, and install TensorFlow Serving
ARG TF_SERVING_BUILD_OPTIONS="--config=release"
RUN echo "Building with build options: ${TF_SERVING_BUILD_OPTIONS}"
ARG TF_SERVING_BAZEL_OPTIONS=""
RUN echo "Building with Bazel options: ${TF_SERVING_BAZEL_OPTIONS}"

RUN bazel build --color=yes --curses=yes \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    /usr/local/bin/

# Build and install TensorFlow Serving API
RUN bazel build --color=yes --curses=yes \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/tools/pip_package:build_pip_package && \
    bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package \
    /tmp/pip && \
    pip --no-cache-dir install --upgrade \
    /tmp/pip/tensorflow_serving_api-*.whl && \
    rm -rf /tmp/pip

# Clean up Bazel cache when done.
RUN bazel clean --expunge --color=yes && \
    rm -rf /root/.cache

#RUN apt-get install tree
