
from keras.models import model_from_json
import os
import sys
import cv2
import argparse
import keras
from keras.utils import np_utils

from PIL import Image
import numpy as np
import json

parser = argparse.ArgumentParser(description='image classifier predictor')
parser.add_argument("-m", "--model", type=str, default=".",
                    help="model location")
parser.add_argument("-f", "--file", type=str, default=None,
                    help="image file")

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def predict_animal(file, model, labelnames):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=labelnames.get(str(label_index))
    print(animal)
    print("The predicted Animal is a %s with accuracy =    %s" % (animal, str(acc)))

def main():
    try:
        args = parser.parse_args()
        model_location = args.model if args.model else "."        
        print("model location: %s" % (model_location))
        # load json and create model
        json_file = open(os.path.join(model_location, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(model_location, "model.h5"))
        print("Loaded model from disk")
        # load label map
        json_file = open(os.path.join(model_location,'label.json'),'r')
        labelnames = json_file.read()
        json_file.close()
        labelnames = json.loads(labelnames)
        
        image_file = args.file
        predict_animal(image_file, model, labelnames)
    except KeyboardInterrupt:
        sys.exit()

if __name__ == '__main__':
    main()