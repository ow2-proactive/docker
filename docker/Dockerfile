# docker build -t activeeon/docker .
# docker run -v /var/run/docker.sock:/var/run/docker.sock -ti --rm activeeon/docker
FROM alpine

COPY --from=library/docker:latest /usr/local/bin/docker /usr/bin/docker
COPY --from=docker/compose:latest /usr/local/bin/docker-compose /usr/bin/docker-compose

RUN apk --update add git less openssh && \
    rm -rf /var/lib/apt/lists/* && \
    rm /var/cache/apk/*

WORKDIR /tmp
