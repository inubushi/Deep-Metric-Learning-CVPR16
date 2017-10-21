# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:56:19 2017

preprocess images

@author: c-morikawa
"""

import conf
import pickle
import img_preprocessor

train_images = pickle.load(open((conf.cache_path + '/Ebay_train.pkl'), 'rb'))
val_images = pickle.load(open((conf.cache_path + '/Ebay_test.pkl'), 'rb'))

# check if all images are readable
# check_images(conf.image_path, train_images); # not implemented yet

# training data
print 'Creating training dataset'
training_images = img_preprocessor.load_cropped_images(conf.image_path, train_images, conf.preprocessing_crop_padding, conf.preprocessing_square_size);

output = open((conf.cache_path + '/training_images_crop15_square256.pkl'), 'wb')
pickle.dump(training_images, output)
output.close()
training_images = []

# validation data
print 'Creating validation dataset'
validation_images = img_preprocessor.load_cropped_images(conf.image_path, val_images, conf.preprocessing_crop_padding, conf.preprocessing_square_size);

output = open((conf.cache_path + '/validation_images_crop15_square256.pkl'), 'wb')
pickle.dump(validation_images, output)
output.close()
validation_images = []