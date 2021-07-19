FROM openjdk:8

########  COMMON SSH CONFIG ########

## Passwordless SSH

#RUN ssh-keygen -q -N "" -t dsa -f /etc/ssh/ssh_host_dsa_key
#RUN ssh-keygen -q -N "" -t rsa -f /etc/ssh/ssh_host_rsa_key
#RUN ssh-keygen -q -N "" -t rsa -f /root/.ssh/id_rsa
#RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys

ADD ssh_config /root/.ssh/config
RUN chmod 600 /root/.ssh/config
RUN chown root:root /root/.ssh/config

RUN echo "Port 2122" >> /etc/ssh/sshd_config

RUN passwd -u root





########  HADOOP ########



# HADOOP
ARG HADOOP_BINARY_ARCHIVE_NAME=hadoop-2.7.0
ARG HADOOP_BINARY_DOWNLOAD_URL=https://archive.apache.org/dist/hadoop/core/hadoop-2.7.0/${HADOOP_BINARY_ARCHIVE_NAME}.tar.gz

# ENV VARIABLES
ENV HADOOP_HOME /usr/local/hadoop
ENV JAVA_HOME /usr/local/openjdk-8
ENV PATH $JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$HADOOP_HOME/bin:$PATH

# Download, uncompress and move all the required packages and libraries to their corresponding directories in /usr/local/ folder. 
RUN apt-get -yqq update && \
apt-get install -yqq nano && \
apt-get install -yqq netcat &&\
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
rm -rf /tmp/* && \
wget -qO - ${HADOOP_BINARY_DOWNLOAD_URL} | tar -xz -C /usr/local/ && \
cd /usr/local/ && \
ln -s ${HADOOP_BINARY_ARCHIVE_NAME} hadoop 


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