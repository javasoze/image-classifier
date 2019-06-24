FROM jjanzic/docker-python3-opencv

COPY requirements.txt .

RUN pip install -r requirements.txt --user

RUN mkdir /image-classifier
ADD train.py /image-classifier
ADD predict.py /image-classifier

WORKDIR /image-classifier