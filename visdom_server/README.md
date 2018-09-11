### Docker build
```
docker build -t activeeon/visdom_server .
```

### Docker image
```
https://hub.docker.com/r/activeeon/visdom_server
```

### Test
```
docker run -it --rm -p 8097:8097 activeeon/visdom_server
```