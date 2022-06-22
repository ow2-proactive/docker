FROM ubuntu:16.04




# install openssh-server, openjdk and wget
RUN apt-get update && apt-get install -y openssh-server openjdk-8-jdk wget vim python

# install hadoop 2.7.2
RUN wget https://github.com/kiwenlau/compile-hadoop/releases/download/2.7.2/hadoop-2.7.2.tar.gz && \
    tar -xzvf hadoop-2.7.2.tar.gz && \
    mv hadoop-2.7.2 /usr/local/hadoop && \
    rm hadoop-2.7.2.tar.gz


# install hbase
RUN wget https://archive.apache.org/dist/hbase/1.4.9/hbase-1.4.9-bin.tar.gz  && \ 
    tar -zxvf hbase-1.4.9-bin.tar.gz && \
    mv hbase-1.4.9 /usr/local/hbase && \
    rm hbase-1.4.9-bin.tar.gz

# install phoenix
RUN wget https://dlcdn.apache.org/phoenix/phoenix-4.16.1/phoenix-hbase-1.4-4.16.1-bin.tar.gz && \
    tar -zxvf phoenix-hbase-1.4-4.16.1-bin.tar.gz && \
    mv phoenix-hbase-1.4-4.16.1-bin /usr/local/phoenix &&\
    rm phoenix-hbase-1.4-4.16.1-bin.tar.gz
    


# set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 
ENV HADOOP_HOME=/usr/local/hadoop 
ENV HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
ENV LD_LIBRARY_PATH=/usr/local/hadoop/lib/native:$LD_LIBRARY_PATH
ENV HBASE_HOME=/usr/local/hbase
ENV CLASSPATH=$CLASSPATH:/usr/local/hbase/lib/*
ENV PHOENIX_HOME /usr/local/phoenix
ENV PATH=$PATH:/usr/local/hadoop/bin:/usr/local/hadoop/sbin:/usr/local/hbase/bin:$PHOENIX_HOME/bin 

# ssh without key
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN apt-get -yqq update && \
apt-get install -yqq nano && \
apt-get install -yqq netcat &&\
apt-get clean 

### ADD TO HBASE DIRECTORY ###
ADD config/hbase-site.xml $HBASE_HOME/conf/hbase-site.xml
ADD config/hdfs-site.xml $HBASE_HOME/conf/hdfs-site.xml
ADD config/hbase-env.sh $HBASE_HOME/conf/hbase-env.sh
ADD config/core-site.xml $HBASE_HOME/conf/core-site.xml

RUN cp $PHOENIX_HOME/phoenix-client-hbase-1.4-4.16.1.jar $HBASE_HOME/lib/phoenix.jar
RUN cp $PHOENIX_HOME/phoenix-server-hbase-1.4-4.16.1.jar $HBASE_HOME/lib/phoenix-server.jar

### ADD TO HADOOP DIRECTORY ###
ADD config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
ADD config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml

### ADD TO ROOT DIRECTORY FOR RUNNING EXAMPLE ###
ADD config/purchases.txt /root/

COPY config/ssh_config /tmp/
RUN mv /tmp/ssh_config ~/.ssh/config

WORKDIR $HBASE_HOME

CMD ["/bin/bash"]