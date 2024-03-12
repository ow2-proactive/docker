# docker build --no-cache -t activeeon/dlm5:latest .
# docker run -it --rm activeeon/dlm5 bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/dlm5 /bin/bash
FROM activeeon/python:3.10

# Update
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt upgrade -y

# Install python libraries
# ------------------------
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN python3 -m pip install numpy==1.26.1
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip cache purge
RUN rm -rf ~/.cache/pip

RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
