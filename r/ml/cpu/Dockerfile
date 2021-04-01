# docker build -t activeeon/r-ml .
# docker run -it --rm activeeon/r-ml bash

# user:pass rstudio:admin
# docker run -it --rm -e PASSWORD=admin -p 8787:8787 activeeon/r-ml

# For more info:
# https://www.rocker-project.org/
# https://hub.docker.com/r/rocker/ml
# https://github.com/rocker-org/ml

FROM rocker/ml

# Update linux
RUN apt-get clean && apt-get update -y -qq

# Update Java
RUN apt install -y openjdk-8-jdk

# Update R
RUN install2.r -s --error \
rJava \
caret \
randomForest \
kernlab \
glmnet \
gbm \
random

# Update Python
RUN pip install py4j

# Fix R
# https://github.com/rstudio/rstudio/issues/2254#issuecomment-372474049
RUN R CMD javareconf

# Update Env
ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV PATH=$PATH:$HOME/bin:$JAVA_HOME/bin:$R_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$R_HOME/lib:$JAVA_HOME/jre/lib/amd64/server
ENV R_LIBS=$R_HOME/lib

# Clean
RUN apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
