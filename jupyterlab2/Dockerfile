# docker build -t activeeon/jupyterlab:latest -f ./Dockerfile .
FROM activeeon/jupyterlab:latest

WORKDIR /opt/notebooks

ENV DEBIAN_FRONTEND noninteractive

# Update APT
# ------------------
RUN apt-get update

# Update Python PIP
# ------------------
RUN pip install --upgrade pip

# Install Python PKGS
# -------------------
RUN python3 -m pip install --upgrade --pre proactive
RUN python3 -m pip install --upgrade "git+https://github.com/ow2-proactive/proactive-jupyter-kernel.git"
RUN python3 -m proactive-jupyter-kernel.install

# Install Proactive Jupyter Notebooks
# -----------------------------------
RUN git clone https://github.com/ow2-proactive/proactive-jupyter-notebooks.git
RUN find ./proactive-jupyter-notebooks -name "*.ipynb" | xargs -r jupyter trust

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

ENTRYPOINT jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.password=$(python -c "from IPython.lib.security import passwd; print(passwd(passphrase='$JUPYTERLAB_PASSWORD', algorithm='sha1'))") $ADDITIONAL_OPTIONS
