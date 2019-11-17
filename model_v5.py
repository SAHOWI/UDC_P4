
#!/usr/bin/python3

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.layers import Lambda, Cropping2D


debug=0
OS='unix'
# OS='win'


### local laptop implementation
# data_dir='C:\\temp\\data\\data\\'
### UDACITY WorkSpace implementation 
data_dir='/home/workspace/CarND-Behavioral-Cloning-P3/data/'





def print_sep():
    print("----------------------------------------")
# end of def: print_set





### read the data first
lines = []
images = []
measurements = []


print_sep()
print("Begin reading data")
print_sep()

with open(data_dir + 'driving_log.csv') as csvfile:
    ### to skip he header line of the CSV file
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

print("Logdata read")
for line in lines:
    ### first fix path in log_file
    source_path = line[0]
    if (OS == 'win'):
        tokens = source_path.split('\\')
    else:
        tokens = source_path.split('/')
        
    filename = tokens[-1]
    if (OS == 'win'):
        local_path = data_dir + filename
    else:
        local_path = data_dir + 'IMG/' + filename
    if (debug == 1):
        print("Local Path = ", local_path)
    #image = cv2.imread(local_path)
    image = cv2.cvtColor(cv2.imread(local_path), cv2.COLOR_BGR2RGB)
    if (debug == 1):
        print(image)
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)


#### Data Augmentation
print_sep()
print("Creating augmented data")
print_sep()

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image=cv2.flip(image,1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

### end of data augmentation

print_sep()
print("Data augmentation done")
print_sep()



print("Data configured")
print_sep()
print("Images                 =", len(images))
print("Measurements           =", len(measurements))
print("Augmented Images       =", len(augmented_images))
print("Augmented Measurements =", len(augmented_measurements))
print_sep()


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

### define our Model

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24,5,5, activation="relu"))

model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))



### compile the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)


model.save('model.h5')


