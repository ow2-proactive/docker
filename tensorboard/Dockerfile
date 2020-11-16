# docker build -t activeeon/tensorboard .
FROM activeeon/tensorflow

RUN mkdir /logs

# Tensorboard
EXPOSE 6006

CMD ["tensorboard", "--logdir", "/logs", "--bind_all"]
