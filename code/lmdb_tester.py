# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:46:38 2017

@author: c-morikawa
"""
"""
import numpy as np
import lmdb
import caffe
import cv2


env = lmdb.open('/home/c-morikawa/git/Deep-Metric-Learning-CVPR16/code/cache/training_set_stanford_multilabel_m128.lmdb', map_size=1e12, readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.height, datum.width, datum.channels)
y = datum.label

print x,y
"""
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
# Wei Yang 2015-08-19
# Source
#   Read LevelDB/LMDB
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#   Plot image
#   ==================
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
#   Creating LMDB in python
#   ==================
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/

#lmdb_file = "/media/c-morikawa/kura/data/train.lmdb"
lmdb_file = "/home/c-morikawa/git/Deep-Metric-Learning-CVPR16/code/cache/train_orig.lmdb"
lmdb_env = lmdb.open(lmdb_file)
print lmdb_env.stat()
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()
 
rec_count = lmdb_txn.stat()['entries']

#for i in range(0, rec_count/2):
#    lmdb_cursor.next()

#myList = [ key for key, _ in lmdb_cursor ]
#print len(myList)

count = 0
for key, value in lmdb_cursor:
    count += 1
    print count
    print key
    #datum.ParseFromString(value)
    #label = datum.label
    #data = caffe.io.datum_to_array(datum)
    #im = data.astype(np.uint8)
    #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    if count % 50000 == 0: 
        print '.',

        #plt.imshow(im)
        #plt.show()
