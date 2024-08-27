# docker build -t activeeon/jupyterlab4 .
# docker run -it --rm activeeon/jupyterlab4 bash
# docker run -it --rm -v $PWD:/home -w /home activeeon/jupyterlab4 /bin/bash
FROM activeeon/python:3.10

WORKDIR /opt/notebooks

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update && apt upgrade -y

# Install required packages
# ------------------------
RUN apt-get install -y \
        apt-utils \
        nodejs \
        gcc \
        g++ \
        make \
        yarn \
        manpages-dev \
        graphviz \
        libgraphviz-dev

# Install python libraries
# ------------------------
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --pre proactive
RUN python3 -m pip install jupyterlab==4.2.5 beakerx==2.4.3 scikit-learn==1.5.1 pandas==2.2.2 ipython==8.26.0

RUN python3 -m pip install "git+https://github.com/ow2-proactive/proactive-jupyter-kernel.git@jupyterlab4"
RUN python3 -m proactive-jupyter-kernel.install

# Install Proactive Jupyter Notebooks
# -----------------------------------
RUN git clone https://github.com/ow2-proactive/proactive-jupyter-notebooks.git
RUN find ./proactive-jupyter-notebooks -name "*.ipynb" | xargs -r jupyter trust

# Install JupyterLab Extensions
# -----------------------------------
RUN python3 -m pip install jupyterlab-git==0.50.1
# RUN python3 -m pip install lckr-jupyterlab-variableinspector==3.2.1

# Clean up
# --------
RUN python3 -m pip cache purge
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose JupyterLab
# -----------------
EXPOSE 8888

ENV JUPYTERLAB_PASSWORD=''
ENV ADDITIONAL_OPTIONS=''

# ENTRYPOINT
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# ENTRYPOINT jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.password=$(python3 -c "from jupyter_server.auth import passwd; print(passwd(passphrase='$JUPYTERLAB_PASSWORD', algorithm='sha1'))") $ADDITIONAL_OPTIONS

# ENTRYPOINT bash
