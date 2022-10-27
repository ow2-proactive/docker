FROM tomcat:9.0.68-jdk8-corretto-al2

RUN rm -rf ${CATALINA_HOME}/webapps/*
RUN curl -o /usr/local/tomcat/webapps/sal.war http://repository.activeeon.com/content/groups/proactive/org/ow2/proactive/scheduling-abstraction-layer/13.1.0-SNAPSHOT/scheduling-abstraction-layer-13.1.0-20221027.125525-9.war

#ENV spring_config_location="/usr/local/tomcat/conf/sal.properties"
ENV EXTERNAL_CONFIG_DIR=${CATALINA_HOME}/conf

ADD  ./scheduling-abstraction-layer.xml ${CATALINA_HOME}/conf/Catalina/localhost/sal.xml
ADD  sal.application.properties ${CATALINA_HOME}/conf/sal.properties

ENV PWS_URL='http://localhost:8080/'
ENV PWS_USERNAME='admin'
ENV PWS_PASSWORD='admin'

ENV DB_USERNAME='root'
ENV DB_PASSWORD=''

ENV DB_DRIVER_CLASSNAME='org.mariadb.jdbc.Driver'
ENV DB_URL='jdbc:mariadb://localhost:3306/proactive'
ENV DB_PLATFORM='org.hibernate.dialect.MySQL5Dialect'

CMD ["catalina.sh","run"]