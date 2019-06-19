# docker build -t activeeon/tensorboard:no-avx .
FROM activeeon/tensorflow

RUN mkdir /logs

# https://github.com/tensorflow/tensorflow/issues/19584
RUN pip uninstall -y tensorflow tensorflow-estimator tensorboard
RUN pip install tensorflow==1.5

# Tensorboard
EXPOSE 6006

CMD ["tensorboard", "--logdir", "/logs"]
