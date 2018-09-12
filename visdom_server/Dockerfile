FROM python:3.6

WORKDIR /root

# Install generic packages
# ------------------------
RUN apt-get update && apt-get install -y \
        wget \
        git \
        unzip \
        nano \
        && apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Visdom (Facebook Research) from source
# ----------------------------------------------
RUN git clone https://github.com/facebookresearch/visdom.git
RUN cd visdom && pip install -e . && easy_install .

RUN rm -rf ~/.cache/pip

ENTRYPOINT ["visdom"]