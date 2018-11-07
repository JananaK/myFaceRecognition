"""
ECE196 Face Recognition Project
Author: W Chen

Use this as a template to:
1. load saved weights for vgg16
2. load test set
3. compute accuracy for test set
"""

import numpy as np
import glob, os, cv2
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical

from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense

TEST_DIR = '/home/ubuntu/myFaceRecognition/data/Test' 
MODEL_PATH = '/home/ubuntu/myFaceRecognition/project_3/weights.h5'
IMG_H, IMG_W, NUM_CHANNELS = 224, 224, 3
MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))
BATCH_SIZE = 16
NUM_CLASSES = 19


def load_data(src_path):
    # under train/val/test dirs, each class is a folder with numerical numbers
    class_path_list = sorted(glob.glob(os.path.join(src_path, '*')))
    image_path_list = []
    for class_path in class_path_list:
        image_path_list += sorted(glob.glob(os.path.join(class_path, '*jpg')))
    num_images = len(image_path_list)
    print '-- This set has {} images.'.format(num_images)
    X = np.zeros((num_images, IMG_H, IMG_W, NUM_CHANNELS))
    Y = np.zeros((num_images, 1))
    # read images and labels
    for i in range(num_images):
        image_path = image_path_list[i]
        label = int(image_path.split('/')[-2])
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (IMG_H, IMG_W)) - MEAN_PIXEL
        X[i, :, :, :] = image
        Y[i, :] = label
    Y = to_categorical(Y, NUM_CLASSES)
    return X, Y


def main():
    # TODO: load model
    base_model = VGG16(include_top=False, weights='imagenet', input_shape = (IMG_H, IMG_W, NUM_CHANNELS))
    print('Model weights loaded.')
    base_out = base_model.output

    x = Flatten()(base_out)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES)(x)  # Simon explained that the number of classes is the number of options we have at the end: aka how many people did we take pics of?
    
    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Build model'
    model.summary()
    model.compile(optimizer = optimizers.SGD(lr=1e-4,momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print 'Compile model'

    # compute test accuracy
    print 'Load test data:'
    X_test, Y_test = load_data(TEST_DIR)
    # TODO: get accuracy
    model.evaluate(x=X_test, y=Y_test, batch_size = BATCH_SIZE, verbose=1, sample_weight=None, steps=BATCH_SIZE)

    return


if __name__ == '__main__':
    main()
