import os
import sys
import numpy as np
import csv
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import decode_predictions
from efficientnet.keras import EfficientNetB7
from efficientnet.keras import center_crop_and_resize, preprocess_input

nr_classes = 10 #number obj of classes to save
class_prob_col = list(['class_'+str(i),'p_'+str(i)] for i in range (nr_classes))
class_prob_col = pd.Series(item for sublist in class_prob_col for item in sublist)
df_colnames = pd.concat([pd.Series(['media_id', 'frame_nr']), class_prob_col])

# loading pretrained model
model = EfficientNetB7() #IMAGENET_WEIGHTS_HASHES not present for EfficientNetL2

"""
Folder structure:
content
 |
 +-- downloaded_videos
 |
 +-- frames
 |  |
 |  +-- folder with frames inside
 |
 +-- efficientnet_emotion_study
 |  |
 |  +-- run_efficient_net.py
"""

frame_folder = os.path.join(os.getcwd(), 'frames') #if only the file is uploaded
final_predictions = list()

for video_folder in os.listdir('frames'):
    media_id = video_folder
    for frame in os.listdir(os.path.join(frame_folder, video_folder)):
        frame_nr = frame[frame.rfind('_')+1:-4]
        img_name = os.path.join(frame_folder, video_folder, frame)

        img =  imread(img_name)
        # preprocess input
        image_size = model.input_shape[1]
        x = center_crop_and_resize(img, image_size=image_size)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)

        y = model.predict(x)
        prediction = decode_predictions(y, top=nr_classes)

        pred_as_list = list()
        for sublist in prediction:
            for pred in sublist:
                pred_as_list.append(pred[1])
                pred_as_list.append(pred[2])

        print(pred_as_list)

        final_predictions.append([media_id, frame_nr] + pred_as_list)

df = pd.DataFrame(final_predictions, columns=df_colnames)
df.to_csv('object_recogniton_values.csv', index=False)
