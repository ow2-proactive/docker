FROM wnameless/oracle-xe-11g

WORKDIR /

COPY instantclient-basic-linux.x64-11.2.0.4.0.zip /

RUN apt-get install zip unzip

RUN mkdir -p /opt/oracle && unzip instantclient-basic-linux.x64-11.2.0.4.0.zip  &&  mv instantclient_11_2/ /opt/oracle/ && rm  instantclient-basic-linux.x64-11.2.0.4.0.zip 

RUN sh -c "echo /opt/oracle/instantclient_11_2 > /etc/ld.so.conf.d/oracle-instantclient.conf"

RUN ldconfig