FROM ubuntu:22.04
LABEL maintainer="zhaomin@u.nus.edu"
RUN apt-get update -y
RUN apt install -y g++-10 gcc-10
RUN apt install -y libgmp3-dev libssl-dev opencl-headers
RUN apt install -y wget build-essential autotools-dev libicu-dev libbz2-dev libboost-all-dev cmake

RUN chmod 777 /tmp

WORKDIR /tmp
RUN wget https://libntl.org/ntl-11.5.1.tar.gz
RUN tar -xvf ntl-11.5.1.tar.gz
WORKDIR /tmp/ntl-11.5.1/src
RUN ./configure SHARED=on
RUN make -j && make install

WORKDIR /tmp
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.bz2
RUN tar -xvf boost_1_75_0.tar.bz2
WORKDIR /tmp/boost_1_75_0
RUN ./bootstrap.sh --prefix=/usr/
RUN ./b2
RUN ./b2 install

COPY DeltaBoost /src/DeltaBoost
RUN apt install -y python3-pip
WORKDIR /src/DeltaBoost
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python
