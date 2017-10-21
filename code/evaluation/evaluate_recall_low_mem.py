# -*- coding: utf-8 -*-
"""
Evaluating recall@k for the online products dataset

Created on Wed Aug 30 14:56:08 2017

@author: c-morikawa
"""
import sys
import scipy.io as io
import numpy as np
import conf

# compute pairwise distance matrix
def distance_matrix(X):
    m = X.shape[0]
    t = np.ones((m,1), 'float32')
    x = np.zeros((m,1), 'float32')
    
    for i in range(0,m):
        cur_vec = X[i,:]
        n = np.linalg.norm(cur_vec)
        x[i] = n * n
    
    # print x
    D = x .dot(t.T) + t.dot(x.T) - 2 * X.dot(X.T)
    
    return D
    
def distance_matrix_row(X,rowID):
    m = X.shape[0]
    t = np.ones((m,1), 'float32')
    x = np.zeros((m,1), 'float32')
    
    # calculate all norms at once
    for i in range(0,m):
        cur_vec = X[i,:]
        n = np.linalg.norm(cur_vec)
        x[i] = n * n
    
    # get value for the current row
    D = (x[rowID]).dot(t.T) + t[rowID].dot(x.T) - 2 * (X[rowID]).dot(X.T)
    
    return D
            
# compute recall@K
def compute_recall_at_K(D, K, image_ids, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        #this_image_id = image_ids[i]
        this_gt_class_idx = class_ids[i];
        this_row = D[i,:];
        # sort the row by ascending order - closest members first
        #[~, inds] = sort(this_row, 'ascend');
        inds = [i[0] for i in sorted(enumerate(this_row), key=lambda x:x[1])]        
        # pick indices for given K
        knn_inds = inds[0:K] # see if this should be K-1
        # pick their class IDs
        knn_class_inds = []
        knn_matches = []
        for j in range(0,K):
            knn_matches.append(image_ids[knn_inds[j]])
            knn_class_inds.append(class_ids[knn_inds[j]])
        
        # count the number of correctly recalled instances
        if this_gt_class_idx in knn_class_inds:
            num_correct = num_correct + 1;
            
        # Testing: let's get a few good matches
        """
        if K >1 and num_correct > 0:
            print 'GT image: ', this_image_id,
            print 'Mismatches: ', knn_matches[:10]
        """
        
    # done
    recall = float(num_correct) / float(num)
    return recall

# compute recall@K
def compute_recall_at_K_low_mem(X, K, image_ids, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        # show something
        if i % 100 == 0:
            print '.',
        #this_image_id = image_ids[i]
        this_gt_class_idx = class_ids[i];
        this_row_sqr = distance_matrix_row(X,i)
        this_row = np.sqrt(abs(this_row_sqr));
        this_row[i] = np.inf
        
        # sort the row by ascending order - closest members first
        #[~, inds] = sort(this_row, 'ascend');
        inds = [i[0] for i in sorted(enumerate(this_row), key=lambda x:x[1])]        
        # pick indices for given K
        knn_inds = inds[0:K] # see if this should be K-1
        # pick their class IDs
        knn_class_inds = []
        knn_matches = []
        for j in range(0,K):
            knn_matches.append(image_ids[knn_inds[j]])
            knn_class_inds.append(class_ids[knn_inds[j]])
        
        # count the number of correctly recalled instances
        if this_gt_class_idx in knn_class_inds:
            num_correct = num_correct + 1;
            
        # Testing: let's get a few good matches
        """
        if K >1 and num_correct > 0:
            print 'GT image: ', this_image_id,
            print 'Mismatches: ', knn_matches[:10]
        """
        
    # done
    recall = float(num_correct) / float(num)
    return recall

## Main 

embedding_dimension = sys.argv[1]

print 'Loading feature matrix...',
name = 'liftedstructsim_softmax_pair_m128_multilabel'
feature_filename = '../clustering/validation_googlenet_feat_matrix_' + name + '_embed' + embedding_dimension + '_baselr_0.0001_gaussian2k.mat'

a = io.loadmat(feature_filename)

features = a['fc_embedding']
# TEST: use only a small part
#features = features[0:5000,:]

dims = features.shape
assert(dims[0] == 60502);
assert(dims[1] == int(embedding_dimension));
print 'done.'

print 'Allocating distance matrix...',
D2 = distance_matrix_row(features, 1)
print 'done.'

# load test data
print 'reading test data...',
image_ids = []
class_ids = []
superclass_ids = []
path_list = []

# open the text file
test_in_file = open(conf.image_path + "/Ebay_test.txt", "r")

# remove the header row
test_in_file.readline()

# loop over the data
for columns in ( row.strip().split() for row in test_in_file ):  
    image_ids.append(int(columns[0]))
    class_ids.append(int(columns[1]))
    superclass_ids.append(int(columns[2]))
    path_list.append(columns[3])

print 'done.'

print 'Calculating recall@k...'
# prepare for calculations
# set diagonal to very high number
#assert(dims[0] == len(class_ids))

num = dims[0]
# use with large matrix
"""
D = np.sqrt(abs(D2));
for i in range(0, num):
    D[i,i] = np.inf;
"""

# calculate recalls at 1, 10, 100 and 1000
for K in [1, 10, 100, 1000]:
    recall = compute_recall_at_K_low_mem(features, K, image_ids, class_ids, num);
    print '   Recall at', K, '=', recall

#done
print 'Finished.'