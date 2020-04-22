# docker build -t activeeon/coco-annotator .
# docker run -d --rm --name coco-annotator -v /var/run/docker.sock:/var/run/docker.sock activeeon/coco-annotator
# docker exec coco-annotator docker-compose down
FROM activeeon/docker

WORKDIR /tmp
#RUN git clone https://github.com/jsbroks/coco-annotator
RUN git clone https://github.com/ow2-proactive/coco-annotator

WORKDIR /tmp/coco-annotator
#RUN docker-compose up -d

#ENTRYPOINT bash -c "docker-compose up -d"
#CMD docker-compose up -d && tail -f /dev/null
CMD docker-compose up
