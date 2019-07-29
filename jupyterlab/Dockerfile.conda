FROM continuumio/anaconda3:2019.03

WORKDIR /root

RUN apt-get update

# Install java
# ------------
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME
RUN apt-get clean

# Install ProActive library
# ------------------------
RUN pip install --no-input -U proactive --upgrade

# Install Graphviz
# ------------------------
RUN apt-get install -y graphviz-dev
RUN conda install -y graphviz

# Install ProActive Jupyter Kernel
# ------------------------
RUN pip install --no-input proactive proactive-jupyter-kernel --upgrade
RUN python -m proactive-jupyter-kernel.install

ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]