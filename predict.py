import csv

import cv2
import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_samples = samples[offset:offset+int(batch_size/2)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if batch_sample[3] == 'steering':
                    continue
                name = '../../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(image_flipped)
                angles.append(center_angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            print("angles", angles)
            yield shuffle(X_train, y_train)

samples = []
with open('../../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

gen = generator(samples, 1000)
#
model = load_model(filepath='model.h5')
#
s = next(gen)
print("shape", s[0])
print("result", s[1])
print("predications", model.predict(s[0]))