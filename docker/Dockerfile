FROM centos:7

# Install gcc 4.8
RUN yum -y install libmpc-devel mpfr-devel gmp-devel zlib-devel bzip2
RUN yum -y groupinstall "Development Tools"

# Install cmake 3.8
WORKDIR /home
RUN curl -O https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
RUN tar zxvf cmake-3.8.0.tar.gz \
	&& cd cmake-3.8.0 \
	&& ./bootstrap --prefix=/usr/local \
	&& make -j4 \
	&& make install

# Install maven, curl and git
RUN yum -y install maven curl
RUN yum -y install http://opensource.wandisco.com/centos/7/git/x86_64/wandisco-git-release-7-2.noarch.rpm && \
	yum -y install git

# clone and build xgboost
WORKDIR /home
RUN git clone --recursive https://github.com/dmlc/xgboost && \
	cd xgboost && \
	make -j4 && \
	cd jvm-packages; mvn package

# clone this project, cp jar files
WORKDIR /home
RUN git clone https://github.com/ShandiLin/xgb_demo.git
RUN mkdir /home/xgb_demo/pyspark_xgb/jars
WORKDIR /home/xgb_demo/pyspark_xgb/jars
RUN cp /home/xgboost/jvm-packages/xgboost4j/target/xgboost4j-0.81-SNAPSHOT.jar ./xgboost4j-0.81.jar
RUN cp /home/xgboost/jvm-packages/xgboost4j-spark/target/xgboost4j-spark-0.81-SNAPSHOT.jar ./xgboost4j-spark-0.81.jar

# spark
WORKDIR /home
RUN curl -O https://archive.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.6.tgz
RUN tar zxvf spark-2.3.1-bin-hadoop2.6.tgz
RUN mv spark-2.3.1-bin-hadoop2.6 /usr/local/

ENV SPARK_HOME /usr/local/spark-2.3.1-bin-hadoop2.6
ENV JAVA_HOME /usr/lib/jvm/java

# install pip & python packages
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
RUN python get-pip.py
RUN pip install pandas===0.23.4 xgboost===0.80 sklearn===0.0

CMD cd /home/xgb_demo