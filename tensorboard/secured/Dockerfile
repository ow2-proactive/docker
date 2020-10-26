# docker build -t activeeon/tensorboard .
FROM activeeon/tensorflow

RUN mkdir /logs

RUN pip install -U pip
RUN pip install -U tensorboard_plugin_profile

RUN apt-get update
RUN apt-get install -y apache2-utils
RUN apt-get install -y nginx

ADD nginx.conf /etc/nginx/sites-available/nginx.conf

ENV PORT=6009
ENV LOGIN="admin"
ENV PASSWORD="admin"

EXPOSE $PORT

COPY docker-entrypoint.sh /
RUN ["chmod", "+x", "/docker-entrypoint.sh"]

ENTRYPOINT ["/docker-entrypoint.sh"]







