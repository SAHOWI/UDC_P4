
#!/usr/bin/python3

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.layers import Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D


### ensure that the workspace will not stop
from workspace_utils import active_session
 
with active_session():
    # do long-running work here
    
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
        correction = 0.2

        for i in (0,2):
            ### first fix path in log_file
            ### and remove leading and trailing white spaces (they are existing in the provided data.zip!!!)
            source_path = line[i].strip()
            if (debug == 1):
                print("file #", i, "= ", source_path)
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
            if (i == 0):
                measurements.append(measurement)
            elif (i == 1):
                steering_left = measurement + correction
                measurements.append(steering_left)
            elif ( i == 2):
                steering_right = measurement - correction
                measurements.append(steering_right)



                
                
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
    print_sep()
    print("Building the Model")
    print_sep()

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25),(0,0))))
    model.add(Conv2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    print_sep()
    print("Done building the Model")
    print_sep()

    ### compile the model
    print_sep()
    print("Compiling the Model")
    print_sep()
    model.compile(optimizer='adam', loss='mse')

    print_sep()
    print("Done compiling the Model")
    print_sep()

    print_sep()
    print("Training the Model")
    print_sep()

    #### this must be changed to model.fit_generator(...)
    #### which requires impementation of a GENERATOR :-)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    print_sep()
    print("Done training the Model")
    print_sep()


    model.save('model.h5')


