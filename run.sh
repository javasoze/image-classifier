docker run \
-it \
--mount type=bind,source="$(pwd)"/train,target=/train \
--mount type=bind,source="$(pwd)"/models,target=/models \
img-classifier bash