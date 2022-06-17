FROM ubuntu:16.04




# install openssh-server, openjdk and wget
RUN apt-get update && apt-get install -y openssh-server openjdk-8-jdk wget vim

# install hadoop 2.7.2
RUN wget https://github.com/kiwenlau/compile-hadoop/releases/download/2.7.2/hadoop-2.7.2.tar.gz && \
    tar -xzvf hadoop-2.7.2.tar.gz && \
    mv hadoop-2.7.2 /usr/local/hadoop && \
    rm hadoop-2.7.2.tar.gz

# set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 
ENV HADOOP_HOME=/usr/local/hadoop 
ENV HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
ENV LD_LIBRARY_PATH=/usr/local/hadoop/lib/native:$LD_LIBRARY_PATH
ENV CLASSPATH=$CLASSPATH:/usr/local/hbase/lib/*
ENV PATH=$PATH:/usr/local/hadoop/bin:/usr/local/hadoop/sbin:/usr/local/hbase/bin 

# ssh without key
RUN ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

RUN apt-get -yqq update && \
apt-get install -yqq nano && \
apt-get install -yqq netcat &&\
apt-get clean 

# HADOOP CONFIG 
RUN chmod +x $HADOOP_HOME/etc/hadoop/hadoop-env.sh
RUN $HADOOP_HOME/etc/hadoop/hadoop-env.sh
# for namenode
ADD hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
ADD core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
# for yarn
ADD yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml
ADD mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml

######## APPLICATION ########
WORKDIR $HADOOP_HOME


CMD ["/bin/bash"]