# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:05:22 2017

@author: c-morikawa
"""

import numpy as np
import lmdb
import caffe

env = lmdb.open('test.lmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        print x.shape
        y = datum.label
        print y

