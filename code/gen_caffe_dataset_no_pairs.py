# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:27:17 2017

write the training data to LMDB

@author: c-morikawa
"""

import conf
import pickle
import random
import numpy as np
import itertools
import math

import Dataset
from caffedata import TrainDB
import lmdb
import cv2


# sampling without replacement - I could not find a python function
def sample_without_replace(ground_set, sample_count):
    selected_samples = []
    selected_count = 0
    while selected_count < sample_count:
        # generate random index
        sample_index = random.randint(0,len(ground_set)-1)
        # we might have to pick more than one
        while ground_set[sample_index] == 0:
            sample_index = random.randint(0,len(ground_set)-1)
        # insert 
        selected_samples.append(sample_index+1) # add 1 because images ids are starting from 1
        # update
        ground_set[sample_index] = 0
        # increment
        selected_count += 1
    return selected_samples


# shaffling samples here    
def shuffle_samples(image_pairs, class_labels):
    shuffled_pairs = []
    shuffled_labels = []
    while len(class_labels) > 0:
        sample_index = random.randint(0,len(class_labels)-1)
        cur_img_pair = image_pairs.pop(sample_index)
        cur_labels = class_labels.pop(sample_index)
        shuffled_pairs.append(cur_img_pair)
        shuffled_labels.append(cur_labels)

    return [np.array(shuffled_pairs), np.array(shuffled_labels)]
    

# generate training pairs
def negative_sampler(num_images, image_ids, num_negs_per_image):
    # originally commented
    # excluding image_ids, 
    # sample (num_negs_per_image * num_views) number of negatives

    num_neg_samples_required = int(num_negs_per_image * len(image_ids));

    # create entire imageset for sampling
    groundset = []
    for i in range(0, num_images):
        groundset.append(1)
    for i in image_ids:
        groundset[i-1] = 0
    
    sample_ids = sample_without_replace(groundset, num_neg_samples_required)
    # add them in a loop
    neg_pair_ids = []
    for i in range(0, len(image_ids)):
        left_image_id = image_ids[i]
        for j in range(0, num_negs_per_image):
            right_image_id = sample_ids[i*num_negs_per_image+j]
            neg_pair_ids.append([left_image_id, right_image_id]);

    # done
    return neg_pair_ids

# generate all training samples here
def get_training_examples_multilabel(mode, batch_size):
    # load dictionaries
    print 'Loading indexes...',
    train_images = pickle.load(open((conf.cache_path + '/Ebay_train.pkl'), 'rb'))
    class_dict = pickle.load(open((conf.cache_path + '/Ebay_dict.pkl'), 'rb'))
    print 'done.'
    
    # set training class count
    class_id_breakpoint = 11318;
    image_list = train_images
    num_images = len(image_list)
    class_list = range(1,class_id_breakpoint+1);
    
    # build reverse lookup for sampling the negative images for forming pairs
    dict_reverse = {}
    keys = class_dict.keys()
    for i in range(0, len(class_dict)):
        this_class_idx = keys[i]
        this_class_img_ids = class_dict[this_class_idx]
        for j in range(0,len(this_class_img_ids)):
            this_img_idx = this_class_img_ids[j];
            dict_reverse[this_img_idx] = this_class_idx;
    
    ## positive samples
    
    # count how many positive examples can be generated exhaustively.
    num_pos_pairs = 0;
    for class_id in class_list:
        this_class_num_images = len(class_dict[class_id]);
        #print 'class: ', class_id, ', num images: ', this_class_num_images
        if this_class_num_images > 1:
            class_pairs = this_class_num_images*(this_class_num_images-1)/2
            #print 'pairs: ', class_pairs
            num_pos_pairs += class_pairs
        else:
            print 'Error: no pairs for this class'            
        # add self pairs, too, to the total
        num_pos_pairs = num_pos_pairs + this_class_num_images;

    print 'Training mode has', num_pos_pairs, 'pos pairs'

    # construct the positive set.
    print 'Creating positive pairs...',
    pos_pairs = []
    pos_class = []
    for class_id in class_list:
        image_ids = class_dict[class_id]
        this_class_num_images = len(image_ids)
        num_combinations = this_class_num_images*(this_class_num_images-1)/2
        #print 'pairs: ', num_combinations
        class_pairs = list(itertools.combinations(image_ids, 2))
        for i in range(0, num_combinations):
            pos_pairs.append(class_pairs[i])
            pos_class.append([class_id, class_id])
        
        # adding self pairs, too
        for i in range(0, this_class_num_images):
            cur_pair = [image_ids[i], image_ids[i]]
            pos_pairs.append(cur_pair)
            pos_class.append([class_id, class_id])

    print 'done.'
    
    ## negative examples
    print 'Creating negative pairs...',
    num_negs_per_image = int(math.ceil(num_pos_pairs / float(num_images)))
    
    neg_pairs = []
    neg_class = []
    
    for class_id in class_list:
        # add an indicator since this takes time
        if class_id % 100 == 0:
            print '\b.', 
        image_ids = class_dict[class_id];        
        sampled_neg_ids = negative_sampler(num_images, image_ids, num_negs_per_image);
        
        # append items in a loop
        for i in range(0, len(sampled_neg_ids)):
            neg_pairs.append(sampled_neg_ids[i])
            class_left = class_id
            class_right = dict_reverse[(sampled_neg_ids[i])[1]]
            neg_class.append([class_left, class_right])
    
    assert(len(neg_pairs) == int(num_negs_per_image * num_images));
    assert(len(neg_pairs) > len(pos_pairs));
    
    print 'done.'
    
    ## Assemble and shuffle

    # make the number of data divisible by the batchsize 256
    # do this by deleting some data from neg pairs b/c more negs than pos
    assert(batch_size % 2 == 0)

    num_total = len(pos_pairs) + len(neg_pairs)
    num_to_delete = int(num_total - math.floor(num_total/float(batch_size/2))*float(batch_size/2));
    
    print 'Deleting', num_to_delete, 'data pairs from negatives for batch divisibility'
    
    # randomly remove so many pairs using python pop
    for i in range(0,num_to_delete):
        rand_index = random.randrange(len(neg_pairs))
        neg_pairs.pop(rand_index)
        neg_class.pop(rand_index)
    
    image_id_pairs = pos_pairs + neg_pairs 
    assert(len(image_id_pairs) % (batch_size/2) == 0)

    labels = pos_class + neg_class
    assert(len(labels) == len(image_id_pairs))
    
    # shuffle
    print 'Shuffling...',
    [shuffled_img_pairs, shuffled_img_labels] = shuffle_samples(image_id_pairs, labels)
    
    # done
    print 'done'
    return [shuffled_img_pairs, shuffled_img_labels]


def serialized_pairs_to_leveldb(images, image_id_pairs_serial, labels_serial, filename):
    print 'Image data:', len(images)
    
    num_examples = len(image_id_pairs_serial)
    assert(num_examples == len(labels_serial))

    # temporary storage for assembling image pairs
    
    # open database file
    db_size = int(1e12)
    db = TrainDB(filename, False,db_size, 128)    
    
    db.writeToDB(images, image_id_pairs_serial, labels_serial)
    
    #done
    db.cleanup()


##--- Main ---##

# set permanent random seed
random.seed(20170201)

# specify the mode manually: 'train' or 'val'
mode = 'train'
# batch size should be a multiple of two; set to 128 according to Matlab code
batch_size = 128

## load images
if mode == 'train':
    print 'Training mode: loading data...',
    images = pickle.load(open(conf.cache_path + '/' + 'training_images_crop15_square256.pkl' , 'rb'))
    assert(len(images)==59551);
else:
    print 'Validation mode: loading data...',
    images = pickle.load(open(conf.cache_path + '/' + 'validation_images_crop15_square256.pkl' , 'rb'))
print 'done.'

image_count = len(images)

## generate pairs
assert(batch_size%2 == 0)
[image_id_pairs, labels] = get_training_examples_multilabel(mode, batch_size)

## Prep dataset for C and Caffe conventions

# get only the pixel information for the images
#images = cat(4, images.img);  # can't do directly, so will use some code
# this needs a huge amount of memory. I will try to skip it and write the rest of the code.
# todo: if really needed, pop elements from images as you go.
image_data = []
while len(images) > 0:
    current_image = images.pop().image
    image_data.append(current_image)
images = None
images = image_data

# verify we are not referring to images that are not in the datasetq
max0 = image_id_pairs.max(axis=0)
max1 = max0.max()
assert(max0.max(axis=0) <= len(image_data));

# images must be (height x width x channels x num)
assert(len(images[0].shape) == 3)
# assume square images
assert(images[0].shape[0] == images[0].shape[1])
# images must have 3 chint num_images = images_size[3];
print 'Image info: width: ', images[0].shape[1], 'height:', images[0].shape[0], 'channels:', images[0].shape[2], 'num:', len(images)
assert(images[0].shape[2] == 3)

# Convert to BGR: not necessary, since we used OpenCV while sampling

# Caffe likes 0-1 labels
labels = np.int32(labels); # might not be needed
# NOTE: this does not make sense
#labels(labels == -1) = 0;

labels_cont = [i for i in range(1, 11318+1)];
temp = set(labels[:,0])
assert(list(temp) == labels_cont)

# Subtract 1 for C indexing
image_id_pairs = (image_id_pairs - 1);

## Do a morph before writing to the database
print 'Reshaping as serial data...',

image_id_pairs_serial = np.zeros(image_id_pairs.shape[0]*2, 'int32');
labels_serial = np.zeros(image_id_pairs.shape[0]*2, 'int32');

# use batch size and reshape
#NOTE: there is no great reason for using batch size here, as used in original code
insert_idx = 0;
lookup_idx = 0;
for i in range (0,len(labels)):
    image_id_pairs_serial[i*2] = image_id_pairs[i,0]
    image_id_pairs_serial[i*2+1] = image_id_pairs[i,1]
    labels_serial[i*2] = labels[i,0]
    labels_serial[i*2+1] = labels[i,1]
    
print 'done.'

## Write to lmdb
# NOTE: we will be writing images to files, and then prepare a text file with labels
# this will then be used by convert_imageset utility in caffe.
print 'Writing to lmdb database...',
if mode == 'train':
    filename = conf.training_set_path_multilabel_m128
elif mode == 'val':
    filename = conf.validation_set_path_multilabel;
else:
    print 'unknown mode ', mode

# save images in a loop: image IDs are from 0 to 59550
for i in range(0, len(images)):
    img_path = conf.path_triplet + '/trg_4_lmdb/img' + str(i) + '.jpg'
    cv2.imwrite(img_path, images[i])
    
## save the text file
output_file = open(conf.path_triplet + '/trg_img_list.txt', "w")
for i in range(0,len(labels_serial)):
    img_index = image_id_pairs_serial[i]
    img_path = 'img' + str(img_index) + '.jpg'
    class_label = labels_serial[i] - 1 # class labels are now starting from 0
    output_file.write("%s %d\n" %(img_path, class_label))
    
output_file.close()
print 'done.'

# that's all, folks!
print 'Completed.'


"""
function gen_caffe_dataset_multilabel_m128(mode, batch_size)

if nargin == 0
    mode = 'train';
    batch_size = 128*1;
end 

conf = config;

%% Load images 
if strcmp(mode, 'train')
    images = load(fullfile(conf.root_path, ...
        'training_images_crop15_square256.mat'), 'training_images');
    images = images.training_images;
    assert(length(images)==59551);
elseif strcmp(mode, 'val')
    images = load(fullfile(conf.root_path, ...
        'validation_images_crop15_square256.mat'), 'validation_images');
    images = images.validation_images;
else
    error('unknown mode %s', mode);
end

%% Pick image pairs
assert(mod(batch_size, 2)==0);
[image_id_pairs, labels] = get_training_examples_multilabel(mode, batch_size);

%% Prep dataset for C and Caffe conventions
images = cat(4, images.img); 

assert(max(image_id_pairs(:)) <= length(images));
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

% Caffe likes 0-1 labels
labels = int32(labels);
labels(labels == -1) = 0;

labels_cont = 1:11318;
assert(isequal(double(unique(labels)), labels_cont'));

% Subtract 1 for C indexing
image_id_pairs = int32(image_id_pairs - 1);

% check the number of paired data is divisible 

%% Do a morph before the mex file

image_id_pairs_serial = int32(zeros(numel(image_id_pairs), 1));
labels_serial = int32(zeros(length(labels)*2, 1));
 
insert_idx = 1;
lookup_idx = 1;
for i = 1:length(labels)
    if mod(i-1, (batch_size/2)) == 0
        image_id_pairs_serial(insert_idx : insert_idx + batch_size-1, :) = ...
            reshape(image_id_pairs(lookup_idx : lookup_idx + batch_size/2-1,:), [], 1);
        
        labels_serial(insert_idx : insert_idx + batch_size-1) = ...
            reshape(labels(lookup_idx : lookup_idx + batch_size/2 -1, :), [], 1);
        
        insert_idx = insert_idx + batch_size;
        lookup_idx = lookup_idx + batch_size/2;
    end
end
assert(length(image_id_pairs_serial) == 2*length(labels));
assert(mod(length(image_id_pairs_serial), batch_size)==0);

%% Write to lmdb
fprintf('Writing level db..\n');
if strcmp(mode, 'train')
    filename = conf.training_set_path_multilabel_m128;
elseif strcmp(mode, 'val')
    filename = conf.validation_set_path_multilabel;
else
    error('unknown mode %s', mode);
end

serialized_pairs_to_leveldb(images, image_id_pairs_serial, labels_serial, filename); 
fprintf('Done writing level db..\n');  
"""