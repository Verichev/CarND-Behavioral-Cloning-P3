import csv

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

print("start")
samples = []
with open('../newdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from scipy import ndimage


def generator(samples, batch_size=64):
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
                path = '../newdata/IMG/'    
                name = path + batch_sample[0].split('/')[-1]
#                 left = path + batch_sample[1].split('/')[-1]
#                 right = path +batch_sample[2].split('/')[-1]
#                 correction = 0.2 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                if steering_center == 0:
                    continue
#                 steering_left = steering_center + correction
#                 steering_right = steering_center - correction
                center_image = ndimage.imread(name)  # cv2.imread(name)
#                 left_image = misc.imread(left)               
#                 right_image = misc.imread(right)
                images.append(center_image)
#                 images.append(left_image)
#                 images.append(right_image)
                angles.append(steering_center)
#                 angles.append(steering_right)
#                 angles.append(steering_left)
                image_flipped = np.fliplr(center_image)
                center_angle_flipped = -steering_center
                images.append(image_flipped)
                angles.append(center_angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
model = load_model(filepath='model.h5')
if model == None:
    print("create model")
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Conv2D(64,3,3, activation='elu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

else:
    print("retrain model")
adam = Adam(lr = 1e-5)
model.compile(loss='mse', optimizer='adam')
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=False,
                             mode='auto')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 2, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1, callbacks=[checkpoint])
model.save('model.h5')
