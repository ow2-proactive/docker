FROM activeeon/dlm3
COPY . /model_as_service
WORKDIR /model_as_service

RUN apt-get clean && apt-get -y update
RUN apt-get -y install nginx

#RUN pip install -r requirements.txt --src /usr/local/src
RUN pip install -r requirements.txt

COPY nginx.conf /etc/nginx

RUN chmod 777 /model_as_service

RUN chmod +x ./start.sh
