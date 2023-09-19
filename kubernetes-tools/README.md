### Overview
**activeeon/k8s-tools** contains a set of tools belonging to the kubernetes ecosystem such as:
- [kubectl](https://kubernetes.io/docs/tasks/tools/) 
- [helm](https://helm.sh/)
- [kops](https://kops.sigs.k8s.io/)
- [AWS cli](https://docs.aws.amazon.com/cli/)


### Build
```
docker build -t activeeon/k8s-tools .
```

### Usage
- docker run --rm -it -v /home/my-user/.aws:/root/.aws activeeon/k8s-tools aws s3 ls
- docker run --rm -it -v /home/my-user/.kube:/root/.kube activeeon/k8s-tools kubectl version
- docker run --rm -it activeeon/k8s-tools kops version
- docker run --rm -it activeeon/k8s-tools helm version

### Source Docker image
- [alpine/k8s](https://hub.docker.com/r/alpine/k8s)
