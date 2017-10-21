# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:20:07 2017

# generate evaluation data

@author: c-morikawa
"""
import conf
import pickle
import cv2

# load images from the pkl file
print 'Validation mode: loading image data...',
images_v = pickle.load(open(conf.cache_path + '/' + 'validation_images_crop15_square256.pkl' , 'rb'))
print 'done.'
# load filenames and dictionary
print 'Loading indexes...',
validation_filenames = pickle.load(open((conf.cache_path + '/Ebay_test.pkl'), 'rb'))
class_dict = pickle.load(open((conf.cache_path + '/Ebay_dict.pkl'), 'rb'))
print 'done.'


# get only the pixel information for the images
#images = cat(4, images.img);  # can't do directly, so will use some code
# this needs a huge amount of memory. I will try to skip it and write the rest of the code.
# todo: if really needed, pop elements from images as you go.
image_data = []
while len(images_v) > 0:
    current_image = images_v.pop().image
    image_data.append(current_image)
images_v = None

image_list = image_data
num_images = len(image_list)

# set test class indices and count
class_list = range(conf.class_id_breakpoint+1, len(class_dict)+1);

# verify that we have the right number of samples
assert(len(image_list) == len(validation_filenames));

print image_list[0].shape

# save images in a loop
print 'Writing images and index...',
for i in range(0, len(image_list)):
    img_path = conf.path_triplet + '/val_4_lmdb/img' + str(i) + '.jpg'
    cv2.imwrite(img_path, image_list[i])
    
## save the text file
output_file = open(conf.path_triplet + '/val_img_list.txt', "w")
for i in range(0,len(image_list)):
    img_path = 'img' + str(i) + '.jpg'
    class_label = 0 # should be 0 according to the CPP code
    output_file.write("%s %d\n" %(img_path, class_label))
    
output_file.close()
print 'done.'


"""
% only validation images
% don't worry about padding. Use small batch size for p5 feat extraction

conf = config;

images_v = load(fullfile(conf.root_path, ...
    'validation_images_crop15_square256.mat'), 'validation_images');
images_v = images_v.validation_images;

images = cat(4, images_v.img);

validation_filenames = {images_v.filename};

assert(length(images) == length(validation_filenames));

% images must be (height x width x channels x num)
assert(ndims(images) == 4); 
% assume square images
assert(size(images, 1) == size(images, 2));
% images must have 3 channels
assert(size(images, 3) == 3);
% Convert to BGR
images = images(:,:,[3 2 1],:);
% Switch width and height
images = permute(images, [2 1 3 4]);

savefilename = conf.validation_set_path;
imageset_to_leveldb(images, savefilename);
fprintf('Done writing level db\n');

save(fullfile(conf.cache_path, 'validation_filenames.mat'), 'validation_filenames');
"""