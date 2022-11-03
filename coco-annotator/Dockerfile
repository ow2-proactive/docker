# docker build -t activeeon/coco-annotator .
# docker run -d --rm --name coco-annotator -v /var/run/docker.sock:/var/run/docker.sock activeeon/coco-annotator
# docker exec coco-annotator docker-compose.yaml down
FROM activeeon/docker

WORKDIR /tmp
RUN git clone https://github.com/ow2-proactive/coco-annotator

WORKDIR /tmp/coco-annotator
CMD SERVICE_HOST_PORT=$SERVICE_HOST_PORT docker-compose up
