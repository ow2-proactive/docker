# Instructions:
# $ docker build -t activeeon/visdom_server .
# $ docker run -it --rm -p 8097:8097 activeeon/visdom_server
# or (for secure connection)
# $ docker run -it --rm -p 8097:8097 -e VISDOM_ENABLE_LOGIN="True" -e VISDOM_USERNAME="admin" -e VISDOM_PASSWORD="admin" activeeon/visdom_server
#
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
RUN git clone https://github.com/ow2-proactive/visdom.git
RUN cd visdom && pip install -e . && easy_install .

RUN rm -rf ~/.cache/pip
RUN mkdir ~/.visdom

ENV VISDOM_HOSTNAME='localhost'
ENV VISDOM_PORT=8097
ENV VISDOM_LOGGING_LEVEL='INFO'
ENV VISDOM_READONLY="False"
ENV VISDOM_ENV_PATH="~/.visdom/"
ENV VISDOM_ENABLE_LOGIN="False"
ENV VISDOM_USERNAME=admin
ENV VISDOM_PASSWORD=admin
ENV VISDOM_USE_ENV_CREDENTIALS=1
ENV VISDOM_COOKIE=visdom
ENV VISDOM_FORCE_NEW_COOKIE="False"
ENV VISDOM_BASE_URL="/"

EXPOSE $PORT

#ENTRYPOINT ["visdom"]

CMD python -m visdom.server \
    --hostname ${VISDOM_HOSTNAME} \
    -port ${VISDOM_PORT} \
    -base_url ${VISDOM_BASE_URL} \
    -env_path ${VISDOM_ENV_PATH} \
    -logging_level ${VISDOM_LOGGING_LEVEL} \
    `if [ "x$VISDOM_READONLY" = "xTrue" ];then echo "-readonly";fi` \
    `if [ "x$VISDOM_FORCE_NEW_COOKIE" = "xTrue" ];then echo "-force_new_cookie";fi` \
    `if [ "x$VISDOM_ENABLE_LOGIN" = "xTrue" ];then echo "-enable_login";fi`
