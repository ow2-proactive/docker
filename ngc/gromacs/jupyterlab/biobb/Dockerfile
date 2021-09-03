# https://mmb.irbbarcelona.org/biobb/availability/tutorials/ubuntu
# docker build -t activeeon/ngc_gromacs_jupyterlab_biobb .
# docker run -it --rm -p 8888:8888 --gpus all activeeon/ngc_gromacs_jupyterlab_biobb
# docker run -it --rm -p 8888:8888 --gpus all -e JUPYTERLAB_PASSWORD=123 -e ADDITIONAL_OPTIONS= activeeon/ngc_gromacs_jupyterlab_biobb

# 20.04.2 LTS (Focal Fossa)
# docker run -it --rm -p 8888:8888 --gpus all nvcr.io/hpc/gromacs:2021 /bin/bash
# FROM nvcr.io/hpc/gromacs:2021

# Ubuntu 18.04.4 LTS
# docker run -it --rm -p 8888:8888 --gpus all nvcr.io/hpc/gromacs:2020.2 /bin/bash
FROM nvcr.io/hpc/gromacs:2020.2

# Ubuntu 18.04.4 LTS
# docker run -it --rm -p 8888:8888 --gpus all nvcr.io/hpc/gromacs:2020 /bin/bash
# FROM nvcr.io/hpc/gromacs:2020

ENV DEBIAN_FRONTEND noninteractive

# RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf
# RUN echo "nameserver 208.67.222.222" >> /etc/resolv.conf
# RUN sed -i '$ d' /etc/apt/sources.list
# RUN sed -i '$ d' /etc/apt/sources.list
RUN apt-get clean && apt-get update -y -qq

RUN apt-get upgrade -y
RUN apt-get install -y software-properties-common apt-utils build-essential nano git
RUN apt-get install -y python3-distutils python3-apt

RUN apt-get install -y python3.7 python3.7-dev
RUN rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN python3 -m pip install --upgrade pip virtualenv

RUN mkdir -p /opt/notebooks
WORKDIR /opt/notebooks

RUN git clone https://github.com/ow2-proactive/biobb_wf_md_setup.git
# RUN cd biobb_wf_md_setup && conda env create -f conda_env/environment.yml
RUN pip install biobb-md==3.6.0 biobb-analysis==3.6.0 biobb_io==3.6.0 biobb-model==3.6.0 biobb-structure-checking==3.8.5
RUN pip install plotly==5.3.1 nglview==3.0.3 simpletraj==0.5
RUN jupyter-nbextension enable --py --user widgetsnbextension
RUN jupyter-nbextension enable --py --user nglview
RUN ln -s /usr/local/lib/python3.7/dist-packages/biobb_structure_checking/check_structure.py /usr/local/bin/check_structure
RUN chmod +x /usr/local/bin/check_structure

RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y yarn
RUN pip install jupyterlab
RUN pip install --upgrade jupyterlab-git

COPY files/* /opt/notebooks/biobb_wf_md_setup/biobb_wf_md_setup/notebooks/

RUN apt-get install -y htop cmake libncurses5-dev libncursesw5-dev
RUN git clone https://github.com/Syllo/nvtop.git
RUN mkdir -p nvtop/build && cd nvtop/build && cmake .. && make && make install
RUN pip install jupyterlab_nvdashboard
# RUN jupyter labextension install jupyterlab-nvdashboard

RUN rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python3
RUN apt-get install -y apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common
RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add -
RUN add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/
RUN mkdir -p /usr/share/man/man1
RUN apt-get update
RUN apt-get install -y adoptopenjdk-8-hotspot
RUN apt-get install -y ca-certificates-java && update-ca-certificates -f
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN add-apt-repository universe
RUN apt-get update
RUN apt-get install -y --fix-missing graphviz libgraphviz-dev

RUN rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3

RUN pip install bash_kernel
RUN python3 -m bash_kernel.install
RUN pip install graphviz pydot
RUN pip install py4j==0.10.8.1
RUN pip install --no-input proactive proactive-jupyter-kernel --upgrade
RUN python3 -m proactive-jupyter-kernel.install
RUN pip install tornado==6.1
RUN git clone https://github.com/ow2-proactive/proactive-jupyter-notebooks.git

RUN apt-get clean all
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8888
ENV JUPYTERLAB_PASSWORD=''
ENV ADDITIONAL_OPTIONS=''

RUN cp /opt/notebooks/biobb_wf_md_setup/biobb_wf_md_setup/notebooks/biobb_MDsetup_tutorial.ipynb .

# CMD jupyter-notebook /opt/notebooks/biobb_wf_md_setup/biobb_wf_md_setup/notebooks/biobb_MDsetup_tutorial.ipynb --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.password=$(python3 -c "from IPython.lib.security import passwd; print(passwd(passphrase='$JUPYTERLAB_PASSWORD', algorithm='sha1'))") $ADDITIONAL_OPTIONS
CMD jupyter lab --notebook-dir=/opt/notebooks --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.password=$(python3 -c "from IPython.lib.security import passwd; print(passwd(passphrase='$JUPYTERLAB_PASSWORD', algorithm='sha1'))") $ADDITIONAL_OPTIONS
