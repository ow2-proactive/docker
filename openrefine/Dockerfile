# docker build -t activeeon/openrefine .
FROM openjdk:8-jre-alpine

#ENV OR_URL https://github.com/OpenRefine/OpenRefine/releases/download/3.2/openrefine-linux-3.2.tar.gz
#ENV OR_URL https://github.com/OpenRefine/OpenRefine/releases/download/3.3/openrefine-linux-3.3.tar.gz
ENV OR_URL https://github.com/OpenRefine/OpenRefine/releases/download/3.4-beta2/openrefine-linux-3.4-beta2.tar.gz

WORKDIR /app

RUN set -xe \
    && apk add --no-cache bash curl grep tar \
    && curl -sSL ${OR_URL} | tar xz --strip 1

VOLUME /data
WORKDIR /data

EXPOSE 3333

ENTRYPOINT ["/app/refine"]
CMD ["-i", "0.0.0.0", "-d", "/data"]
