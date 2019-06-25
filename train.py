import os
import sys
import argparse
import cv2
import keras
import json
from keras.utils import np_utils

from PIL import Image
import numpy as np

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# supported extensions
ext = {".jpeg", ".jpg", ".png"}
parser = argparse.ArgumentParser(description='image classifier trainer')
parser.add_argument("-d", "--data", type=str, default=None, required=True,
                    help="training data location")
parser.add_argument("-o", "--output", type=str, default=".",
                    help="model output location")

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def main():
    try:
        data=[]
        labels=[]
        labelnames={}
        
        args = parser.parse_args()
        top = args.data
        model_location = args.output
        print("training data location: %s" % (top))
        print("output model locaiton: %s" % (model_location))        
        count = 0
        for f in os.listdir(top):
            pathname = os.path.join(top, f)
            if os.path.isdir(pathname):        
                for img in os.listdir(pathname):
                    extension = os.path.splitext(img)[1]            
                    if extension in ext:
                        imgfile=os.path.join(pathname, img)
                        resized_image = convert_to_array(imgfile)
                        data.append(resized_image)
                        labels.append(count)
                        labelnames[str(count)]=f
                count += 1

        with open(os.path.join(model_location,"label.json"), "w") as json_file:
            json_file.write(json.dumps(labelnames))

        animals=np.array(data)
        labels=np.array(labels)

        #np.save("animals",animals)
        #np.save("labels",labels)

        s=np.arange(animals.shape[0])
        np.random.shuffle(s)
        animals=animals[s]
        labels=labels[s]

        num_classes=len(np.unique(labels))
        data_length=len(animals)

        (x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        train_length=len(x_train)
        test_length=len(x_test)

        (y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

        #One hot encoding
        y_train=keras.utils.to_categorical(y_train,num_classes)
        y_test=keras.utils.to_categorical(y_test,num_classes)

        #make model
        model=Sequential()
        model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(500,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(count,activation="softmax"))
        model.summary()

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['accuracy'])

        model.fit(x_train,y_train,batch_size=50,epochs=100,verbose=1)

        score = model.evaluate(x_test, y_test, verbose=1)
        print('\n', 'Test accuracy:', score[1])

        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(model_location,"model.json"), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(model_location,"model.h5"))
        print("Saved model to disk")
    except KeyboardInterrupt:
        sys.exit()

if __name__ == '__main__':
    main()
