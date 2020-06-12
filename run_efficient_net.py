import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import decode_predictions
from efficientnet.keras import EfficientNetB7
from efficientnet.keras import center_crop_and_resize, preprocess_input

# test image
image = imread('misc/panda.jpg')

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# loading pretrained model
model = EfficientNetB7() #IMAGENET_WEIGHTS_HASHES not present for EfficientNetL2

# preprocess input
image_size = model.input_shape[1]
x = center_crop_and_resize(image, image_size=image_size)
x = preprocess_input(x)
x = np.expand_dims(x, 0)

# make prediction and decode
y = model.predict(x)
decode_predictions(y)
