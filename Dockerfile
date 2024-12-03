FROM openjdk:8-jdk-slim
COPY --from=python:3.7 / /

ENV ENV <ENV>
ENV IQ_USER iquartic
ENV IQ_UID 1000
ENV HOME /home/${IQ_USER}

ENV PYSPARK_PYTHON=python3.7
ENV PYSPARK_DRIVER_PYTHON=python3.7

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y unzip && \
    apt-get install -y build-essential
#    apt-get install -y python-numpy
RUN apt-get update -qq

WORKDIR ${HOME}

COPY app/ ${HOME}

RUN pip3 install --no-cache-dir pyspark==3.1.1 \
    spark-nlp==3.1.1

# install jsl spark nlp 
RUN pip3 install spark-nlp-jsl==3.1.1 --extra-index-url https://pypi.johnsnowlabs.com/<JSL_SPARK_LICENSE_SECRET_VERSION_3_1_1>

RUN wget https://pypi.johnsnowlabs.com/<JSL_SPARK_LICENSE_SECRET_VERSION_3_1_1>/spark-nlp-jsl-3.1.1.jar

RUN wget https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-3.1.1.jar

RUN pip3 install -r requirements.txt

RUN pip3 install spacy==3.1.4 spacy-legacy==3.0.8
RUN pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV SPARK_NLP_LICENSE <JSL_SPARK_NLP_LICENSE> 

ENV JAVA_OPTS='-Xmx30g'

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

