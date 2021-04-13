FROM ubuntu

# install required packages
RUN apt-get update -y \
   && apt-get install make -y \
   && apt-get install libglu-dev libxinerama-dev libxi-dev -y

COPY vmd-1.9.4a48.bin.LINUXAMD64-CUDA102-OptiX650-OSPRay185.opengl.tar.gz /opt

RUN cd /opt &&  tar xfz vmd-1.9.4a48.bin.LINUXAMD64-CUDA102-OptiX650-OSPRay185.opengl.tar.gz

RUN ln -s  /opt/vmd-1.9.4a48 /opt/vmd

RUN cd /opt/vmd && ./configure LINUXAMD64

RUN cd /opt/vmd/src && make install

CMD vmd



