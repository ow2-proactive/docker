# docker build -t activeeon/maas_ml:gpu .
FROM activeeon/rapidsai:latest

COPY nginx.conf /model_as_service/nginx.conf
COPY start.sh /model_as_service/start.sh
COPY uwsgi.ini /model_as_service/uwsgi.ini
COPY templates/ /model_as_service/templates/
COPY static/ /model_as_service/static/
COPY assets/ /model_as_service/assets/
COPY dash_utils.py /model_as_service/dash_utils.py
COPY requirements.txt /model_as_service/requirements.txt
RUN mkdir -p /model_as_service/models/
WORKDIR /model_as_service

RUN apt-get clean && apt-get -y update
RUN apt-get -y install nginx
RUN apt-get install tree

RUN pip install -r requirements.txt

COPY nginx.conf /etc/nginx

RUN chown -R www-data:www-data /var/www

RUN chmod -R 776 /model_as_service

RUN chmod +x ./start.sh
