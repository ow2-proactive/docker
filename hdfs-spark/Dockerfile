FROM openjdk:8


######## PYSPARK / PY4J ########

RUN apt-get clean && apt-get update -y -qq
RUN apt-get install -yqq python-pip
RUN pip install py4j pyspark==2.4.2


######## SPARK / HADOOP ########

# SPARK
ARG SPARK_BINARY_ARCHIVE_NAME=spark-2.4.2-bin-without-hadoop
ARG SPARK_BINARY_DOWNLOAD_URL=https://archive.apache.org/dist/spark/spark-2.4.2/${SPARK_BINARY_ARCHIVE_NAME}.tgz

# HADOOP
ARG HADOOP_BINARY_ARCHIVE_NAME=hadoop-2.7.0
ARG HADOOP_BINARY_DOWNLOAD_URL=https://archive.apache.org/dist/hadoop/core/hadoop-2.7.0/${HADOOP_BINARY_ARCHIVE_NAME}.tar.gz

# ENV VARIABLES
ENV HADOOP_HOME /usr/local/hadoop
ENV SPARK_HOME /usr/local/spark
ENV JAVA_HOME /usr/local/openjdk-8
ENV PATH $JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$HADOOP_HOME/bin:$PATH

# Download, uncompress and move all the required packages and libraries to their corresponding directories in /usr/local/ folder. 
RUN apt-get -yqq update && \
apt-get install -yqq vim screen tmux && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
rm -rf /tmp/* && \
wget -qO - ${SPARK_BINARY_DOWNLOAD_URL} | tar -xz -C /usr/local/ && \
wget -qO - ${HADOOP_BINARY_DOWNLOAD_URL} | tar -xz -C /usr/local/ && \
cd /usr/local/ && \
ln -s ${SPARK_BINARY_ARCHIVE_NAME} spark && \
ln -s ${HADOOP_BINARY_ARCHIVE_NAME} hadoop && \
cp spark/conf/log4j.properties.template spark/conf/log4j.properties && \
mkdir /tmp/spark-events && \
sed -i -e s/WARN/ERROR/g spark/conf/log4j.properties && \
sed -i -e s/INFO/ERROR/g spark/conf/log4j.properties

# HADOOP CONFIG 
RUN chmod +x $HADOOP_HOME/etc/hadoop/hadoop-env.sh
RUN $HADOOP_HOME/etc/hadoop/hadoop-env.sh
# for namenode
ADD hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
# for datanode
ADD hdfs-site.xml.template $HADOOP_HOME/etc/hadoop/hdfs-site.xml.template
# for namenode & datanode
ADD core-site.xml.template $HADOOP_HOME/etc/hadoop/core-site.xml.template

#ENV SPARK_DIST_CLASSPATH "$HADOOP_HOME/etc/hadoop:$HADOOP_HOME/share/hadoop/common/lib/:$HADOOP_HOME/share/hadoop/common/:$HADOOP_HOME/share/hadoop/hdfs:$HADOOP_HOME/share/hadoop/hdfs/lib/:$HADOOP_HOME/share/hadoop/hdfs/:$HADOOP_HOME/share/hadoop/yarn/lib/:$HADOOP_HOME/share/hadoop/yarn/:$HADOOP_HOME/share/hadoop/mapreduce/lib/:$HADOOP_HOME/share/hadoop/mapreduce/:$HADOOP_HOME/contrib/capacity-scheduler/.jar:$HADOOP_HOME/share/hadoop/tools/lib/"

ENV SPARK_DIST_CLASSPATH "/usr/local/hadoop-2.7.0/etc/hadoop:/usr/local/hadoop-2.7.0/share/hadoop/common/lib/*:/usr/local/hadoop-2.7.0/share/hadoop/common/*:/usr/local/hadoop-2.7.0/share/hadoop/hdfs:/usr/local/hadoop-2.7.0/share/hadoop/hdfs/lib/*:/usr/local/hadoop-2.7.0/share/hadoop/hdfs/*:/usr/local/hadoop-2.7.0/share/hadoop/yarn/lib/*:/usr/local/hadoop-2.7.0/share/hadoop/yarn/*:/usr/local/hadoop-2.7.0/share/hadoop/mapreduce/lib/*:/usr/local/hadoop-2.7.0/share/hadoop/mapreduce/*:/usr/local/hadoop/contrib/capacity-scheduler/*.jar" 


######## APPLICATION ########

CMD ["/bin/bash"]
