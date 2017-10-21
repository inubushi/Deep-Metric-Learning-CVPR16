# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:05:15 2017

# preprocess images in the Stanford objects dataset

@author: c-morikawa
"""
import pickle
import conf

image_ids = []
class_ids = []
superclass_ids = []
path_list = []

## training
    
# open the text file
train_in_file = open(conf.image_path + "/Ebay_train.txt", "r")

# remove the header row
train_in_file.readline()

# loop over the data
for columns in ( row.strip().split() for row in train_in_file ):  
    image_ids.append(int(columns[0]))
    class_ids.append(int(columns[1]))
    superclass_ids.append(int(columns[2]))
    path_list.append(columns[3])

# create dictionary entries
class_dict = {}
train_images = [] 
test_images = []

for i in range(0,len(image_ids)):
    imageid = image_ids[i]
    classid = class_ids[i]
    filename = path_list[i]
    
    print i+1, '/', len(image_ids), ' ', classid, ' ',filename
    
    # append to the image list
    train_images.append(filename)
    
    # add to the dictionary
    class_dict.setdefault(classid,[])
    class_dict[classid].append(imageid)
    
# save
output = open((conf.cache_path + '/Ebay_train.pkl'), 'wb')
pickle.dump(train_images, output)
output.close()

## testing
image_ids = []
class_ids = []
superclass_ids = []
path_list = []
    
# open the text file
train_in_file = open(conf.image_path + "/Ebay_test.txt", "r")

# remove the header row
train_in_file.readline()

# loop over the data
for columns in ( row.strip().split() for row in train_in_file ):  
    image_ids.append(int(columns[0]))
    class_ids.append(int(columns[1]))
    superclass_ids.append(int(columns[2]))
    path_list.append(columns[3])

# create list of dictionary entries


for i in range(0,len(image_ids)):
    imageid = image_ids[i]
    classid = class_ids[i]
    filename = path_list[i]
    
    print i+1, '/', len(image_ids), ' ', classid, ' ',filename
    
    # append the filename
    test_images.append(filename)
    
    # add to the dictionary
    class_dict.setdefault(classid,[])
    class_dict[classid].append(imageid)
    
# save
output = open((conf.cache_path + '/Ebay_test.pkl'), 'wb')
pickle.dump(test_images, output)
output.close()

# save the class dictionary now
output = open((conf.cache_path + '/Ebay_dict.pkl'), 'wb')
pickle.dump(class_dict, output)
output.close()

