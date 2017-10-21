#!/usr/bin/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import caffe
import scipy.io as io
import sys
import pickle
import cv2

assert len(sys.argv)==3, "Incorrect no. of inputs. Provide embedding dimension and baselr."  
embedding_dimension = int(sys.argv[1])
baselr = sys.argv[2]
print 'Embedding dim: %d' % embedding_dimension
#embedding_dimension = 64

MODEL_FILE = 'model/extract_googlenet_ebay_feature_embed%d.prototxt' % (embedding_dimension)

#PRETRAINED = 'snapshot_product_m128_multilabel_embed%d_baselr_%s_iter_20000.caffemodel' % (embedding_dimension, baselr)
# ours
#PRETRAINED = 'model/snapshot_product_128_iter_300000.caffemodel'
# theirs
PRETRAINED = '/media/c-morikawa/kura/models/Deep-Metric-Learning-CVPR16/pretrained_models/OnlineProducts/snapshot_ebay_googlenet_finetune_liftedstructsim_softmax_pair_m128_multilabel_embed128_baselr_1E4_iter_20000.caffemodel'

MEAN_FILE = 'cache/imagenet_mean.binaryproto'

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

import lmdb
LMDB_FILENAME = '/media/c-morikawa/kura/data/メルカリ/Deep-Metric-Learning-CVPR16/val.lmdb'
lmdb_env = lmdb.open(LMDB_FILENAME)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

num_imgs = 0
for key in lmdb_cursor:
    num_imgs += 1
print 'Num images: %d' %num_imgs

batchsize = 26
num_batches = num_imgs / batchsize 
print 'Num batches: %d' %num_batches

# Store fc features for all images
fc_feat_dim = embedding_dimension

feat_matrix = np.zeros((num_imgs, fc_feat_dim), dtype=np.float32)
filename_list = []

filename_idx = 0
for batch_id in range(num_batches): # should be num_batches! updated for tessting 20171012
    if batch_id %100 ==0:    
        print batch_id    
    batch = net.forward()
    fc = net.blobs['fc_embedding'].data.copy()
    for i in range(batchsize):
        feat_matrix[filename_idx+i, :] = fc[i,:]
        
    filename_idx += batchsize


a = {}
a['fc_embedding'] = feat_matrix
pickle.dump(feat_matrix, open('feature_matrix.pkl','wb'))
# save
# ours
#io.savemat('clustering/validation_googlenet_feat_matrix_liftedstructsim_softmax_pair_m128_multilabel_embed%d_baselr_%s_gaussian2k.mat' % (embedding_dimension, baselr), a)
# theirs
#io.savemat('clustering/validation_googlenet_feat_matrix_liftedstructsim_softmax_pair_m128_multilabel_embed%d_baselr_%s_gaussian2k_orig.mat' % (embedding_dimension, baselr), a)
print feat_matrix[0,:]

print "all zeros?: ", np.allclose(feat_matrix, 0)
