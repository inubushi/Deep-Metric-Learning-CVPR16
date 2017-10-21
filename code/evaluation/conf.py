# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:33:42 2017

configuration for object matching

@author: c-morikawa
"""
## Directories
root_path = '/home/c-morikawa/git/Deep-Metric-Learning-CVPR16/code'
cache_path = '/home/c-morikawa/git/Deep-Metric-Learning-CVPR16/code/cache'
image_path = '/media/c-morikawa/kura/data/Stanford_Online_Products'

## Training parameters
preprocessing_crop_padding = 15
preprocessing_square_size = 256
preprocessing_num_to_load = 255
preprocessed_image_file = cache_path + '/training_images.mat'

# number of training samples, used as a breakpoint            
class_id_breakpoint = 11318;

path_triplet = '/home/c-morikawa/git/Deep-Metric-Learning-CVPR16/code/cache';

# for multilabel pairs batchsize = 128
training_set_path_multilabel_m128 = path_triplet + '/training_set_stanford_multilabel_m128.lmdb'

# for multilabel pairs batchsize = 128*2 = 256
training_set_path_multilabel_m256 = path_triplet + '/training_set_stanford_multilabel_m256.lmdb'

# for multilabel pairs batchsize = 128*3 = 384
training_set_path_multilabel_m384 = path_triplet + '/training_set_cars196_multilabel_m384.lmdb'

# for debugging
training_imageset_path = path_triplet + '/training_imageset_cars196.lmdb'

training_set_path_triplet = path_triplet + '/training_set_triplet.lmdb'
validation_set_path_triplet = path_triplet + '/validation_set_triplet.lmdb'
validation_set_path = path_triplet + '/validation_set_cars196.lmdb'