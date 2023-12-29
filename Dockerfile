FROM ubuntu:22.04
RUN apt-get -y update && apt-get install -y git python3.10 pip
RUN pip install tensorflow==2.10.0 scikit-learn matplotlib pandas
RUN git clone https://github.com/gaanno/proyecto-ml
WORKDIR /proyecto-ml
