FROM activeeon/yarn:latest

########  HBASE ########
# https://dlcdn.apache.org/hbase/2.4.8/hbase-2.4.8-bin.tar.gz


ARG HBASE_BINARY_DOWNLOAD_URL=https://dlcdn.apache.org/hbase/2.4.8/hbase-2.4.8-bin.tar.gz

RUN wget ${HBASE_BINARY_DOWNLOAD_URL} && \
tar -zxvf hbase-2.4.8-bin.tar.gz  && \
rm -rf hbase-2.4.8-bin.tar.gz && \
mv hbase-2.4.8 /usr/local/hbase && \
cd /usr/local/


### ENVIRONNMENT VARIABLES ###
ENV HBASE_HOME /usr/local/hbase
ENV PATH $PATH:$HBASE_HOME/bin


### ADD TO HBASE DIRECTORY ###
ADD config/hbase-site.xml $HBASE_HOME/conf/hbase-site.xml
ADD config/hdfs-site.xml $HBASE_HOME/conf/hdfs-site.xml
ADD config/hbase-env.sh $HBASE_HOME/conf/hbase-env.sh
ADD config/core-site.xml $HBASE_HOME/conf/core-site.xml

### ADD TO HADOOP DIRECTORY ###
ADD config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
ADD config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml

WORKDIR $HBASE_HOME

CMD ["/bin/bash"]