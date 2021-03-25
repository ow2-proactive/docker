# docker build -t activeeon/r-ml:gpu .
# docker run -it --rm --gpus all --privileged activeeon/r-ml:gpu bash
# docker run -it --rm --gpus all --privileged activeeon/r-ml:gpu nvidia-smi

# user:pass rstudio:admin
# docker run -it --rm --gpus all --privileged -e PASSWORD=admin -p 8787:8787 activeeon/r-ml:gpu

# For more info:
# https://www.rocker-project.org/
# https://hub.docker.com/r/rocker/ml
# https://github.com/rocker-org/ml

FROM rocker/ml-gpu

# Update linux
RUN apt-get clean && apt-get update -y -qq
RUN apt install -y liblzma-dev libbz2-dev

# Update Java
# https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-ubuntu-20-04
RUN apt install -y default-jre
RUN apt install -y default-jdk

# Update R
RUN install2.r -s --error \
rJava \
caret

# Update Python
RUN python -V
RUN which python
RUN pip -V
RUN pip install py4j

# Clean
RUN apt-get autoclean && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
