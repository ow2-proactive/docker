[OpenRefine](http://openrefine.org/) is a free, open source power tool for working with messy data and improving it.

### Docker build
```
docker build -t activeeon/openrefine .
```

### Docker image
```
https://hub.docker.com/r/activeeon/openrefine
```

### Docker run
```
docker run -p 3333:3333 activeeon/openrefine:latest

```

point your browser on host machine to http://localhost:3333 (or on any machine within your network)


### Example for customized run command
```
docker run --rm -p 80:3333 -v /home/user/data:/data:z activeeon/openrefine:latest -i 0.0.0.0 -d /data -m 4G

```

* automatically remove docker container when it exits (`--rm`)
* publish internal port 3333 to host port 80 (`-p 80:3333`)
* let OpenRefine read and write data in host directory
  * mount host path /home/felix/refine to container path /data (`-v /home/felix/refine:/data:z`)
  * set OpenRefine workspace to /data (`-d /data`)
* pin docker tag 3.2 (i.e. OpenRefine version) (`:3.2`)
* set Openrefine to be accessible from outside the container, i.e. from host (`-i 0.0.0.0`)
* increase java heap size to 4G (`-m 4g`)

### See also

* Command line interface for OpenRefine: [openrefine-client](https://github.com/opencultureconsulting/openrefine-client/#docker)
* Linux Bash script to run OpenRefine in batch mode (import, transform, export): [openrefine-batch-docker.sh](https://github.com/opencultureconsulting/openrefine-batch/#docker)
* https://github.com/opencultureconsulting/openrefine-docker
