# image-classifier
image classifier based on keras and tensorflow

### Prerequisites
* Docker (https://docs.docker.com/v17.12/install/)

## Install
```
docker pull javasoze/image-classifier
```

## Creating trainning data

create a directory with training data using directory names as labels, e.g. using `training` as the top-level directory name:

```
train
  /dog
     dog-img1.jpeg
     dog-img2.jpeg
     ...
  /cat
     cat-img1.jpeg
     img2.png
     img3.jpeg
```

## Training

create a directory to store models:
```
mkdir models
```

run classifier virtual machine:
```
docker run \
-it \
--mount type=bind,source="$(pwd)"/train,target=/train \
--mount type=bind,source="$(pwd)"/models,target=/models \
javasoze/image-classifier bash
```

once inside virtual machine's shell:
```
python train.py -d /train -o /models
```

Model will be output to models directory

## Prediction

Given a new image to predict, e.g. myimage.jpeg:

Let's first copy it to the `models` directory we created earlier

again, run the classifier virtual machine:
```
docker run \
-it \
--mount type=bind,source="$(pwd)"/train,target=/train \
--mount type=bind,source="$(pwd)"/models,target=/models \
javasoze/image-classifier bash
```
once inside virtual machine's shell:
```
python predict.py -m /models -f /models/myimage.jpeg
```
