FROM alpine/k8s:1.28.2

RUN curl -Lo kops https://github.com/kubernetes/kops/releases/download/$(curl -s https://api.github.com/repos/kubernetes/kops/releases/latest | grep tag_name | cut -d '"' -f 4)/kops-linux-amd64
RUN chmod +x kops
RUN mv kops /usr/local/bin/kops

RUN apk update

RUN apk add jq \
    openssh \
    groff \
    less

RUN apk add --no-cache aws-cli
