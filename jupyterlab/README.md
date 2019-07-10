### `pip` based version

#### Docker build

```aidl
docker build -t activeeon/jupyterlab:pip -f ./Dockerfile.pip .
```

#### Docker run

```aidl
docker run -it -p 8888:8888 activeeon/jupyterlab:pip
```

### `conda` based version

#### Docker build

```aidl
docker build -t activeeon/jupyterlab:conda -f ./Dockerfile.conda .
```

#### Docker run

```aidl
docker run -it -p 8888:8888 activeeon/jupyterlab:conda
```

## Docker images

```
https://hub.docker.com/r/activeeon/jupyterlab
```
