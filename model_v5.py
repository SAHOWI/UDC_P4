
#!/usr/bin/python3

import csv
import cv2

data_dir='C:\\temp\\data\\data\\'





def print_sep():
    print("----------------------------------------")
# end of def: print_set

### read the data first
lines = []
images = []
measurements = []

with open(data_dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("Logdata read")
for line in lines:
    source_path = line[0]
    tokens = source_path.split('\\')
    filename = tokens[-1]
    # local_path = 'somedir' + filename
    local_path = data_dir + filename ### actually replaces A by A in my case ;-)
    image = cv2.imread(local_path)
    images.append(image)
    measurement=line[3]
    measurements.append(measurement)
print("Data confi8gured")
print_sep()
print("Images       =", len(images))
print("Measurements =",  len(measurements))
print_sep()


