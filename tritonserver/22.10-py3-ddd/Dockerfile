# docker build -t activeeon/tritonserver:22.10-py3-ddd .
FROM nvcr.io/nvidia/tritonserver:22.10-py3

# Update Python PIP
RUN python3 -m pip install --upgrade pip

# Install Python Packages
COPY requirements.txt /opt/requirements.txt
RUN python3 -m pip install -r /opt/requirements.txt

# Update APT
RUN apt-get update -y

# Install utils
RUN apt-get install htop net-tools iputils-ping jq -y

# Install Redis
RUN apt install -y lsb-release curl gpg
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
RUN apt-get update
RUN apt-get install -y redis
# RUN redis-server --daemonize yes && sleep 1 && redis-cli ping

# Install Visual Studio Code Server
# RUN wget -O- https://aka.ms/install-vscode-server/setup.sh | sh
ENV CODESERVER_VERSION=4.92.2
RUN curl -fOL https://github.com/coder/code-server/releases/download/v$CODESERVER_VERSION/code-server_${CODESERVER_VERSION}_amd64.deb
RUN dpkg -i code-server_${CODESERVER_VERSION}_amd64.deb
RUN rm -rf *.deb
RUN mkdir -p /root/.config/code-server
# COPY code-server/config.yaml /root/.config/code-server/config.yaml
# Install extensions
RUN code-server --install-extension ms-python.python
#RUN code-server --install-extension ms-python.vscode-pylance
#RUN code-server --install-extension ms-vscode.remote-explorer
RUN code-server --install-extension oderwat.indent-rainbow
#RUN code-server --install-extension DavidKol.fastcompare
RUN code-server --install-extension moshfeu.compare-folders
#RUN code-server --install-extension Bito.Bito

# Install Node.js, NPM, Yarn
# ENV NODE_MAJOR=20
# RUN mkdir -p /etc/apt/keyrings
# RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
# RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list
# RUN apt-get update
# RUN apt-get install nodejs -y
# RUN npm install -g npm@10.2.0
# RUN npm install --global yarn

# Install npm
RUN apt-get install -y npm
RUN npm install -g nodemon

# Copy triton proxy server
COPY triton_proxy/ /opt/triton_proxy/
RUN cd /opt/triton_proxy/ && npm init -y && npm install express http-proxy body-parser
# cd /opt/triton_proxy/ && node triton_proxy.js

# Copy rest api server script
COPY api_server/ /opt/api_server/

# Clean cache
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PASSWORD=
RUN export PASSWORD

# Expose ports
# (code-server, api-server, http, grpc, metrics, proxy-server, redis-server)
EXPOSE 9999 9000 8000 8001 8002 8080 6379

# Configure default container execution command
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD []
