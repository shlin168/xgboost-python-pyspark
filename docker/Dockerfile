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

CMD ["/bin/bash"]