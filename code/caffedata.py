# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:23:43 2017

# Database object for storing training data

@author: c-morikawa
"""
import numpy as np
import lmdb
import caffe

DB_KEY_FORMAT = "{:0>10d}"

class TrainDB:
    
    def __init__(self, db_path, use_leveldb, count, batch_size):
        if use_leveldb:
            # not used
            print ' Error: LevelDB not supported!'
        else:
            print 'Opening lmdb database'
            self.map_size = count
            self.env = lmdb.open(db_path, map_size=self.map_size)

    def writeToDB(self, images, image_id_pairs_serial, labels_serial):
        # making a copy of the image
        #serialized_pair_buf = np.zeros(images[0].shape, np.uint8)
        """
        curr_idx = 0
        commit_size = 1000
        with self.env.begin(write=True) as in_txn:
            for i in range(0,1000):
                d, l = images[i], labels_serial[i] # wrong order, just for testing
                im_dat = caffe.io.array_to_datum(d.astype(float), label=int(l))
                key = DB_KEY_FORMAT.format(curr_idx)
                in_txn.put(key, im_dat.SerializeToString())
                curr_idx += 1
        self.env.close()
        """        
        with self.env.begin(write=True) as txn:
            # write data in a loop    
            #for i in range(0, len(image_id_pairs_serial)):      
            for i in range(0, 10): 
                # an indicator, in case this takes long        
                if i%1000 == 0:
                    print i, '.',
        
                img_id = image_id_pairs_serial[i]
                if img_id > 0 and img_id <= len(images):
                    serialized_pair_buf = images[img_id-1]
            
                    datum = caffe.proto.caffe_pb2.Datum()
            
                    # set image
                    datum.channels = serialized_pair_buf.shape[2]
                    datum.height = serialized_pair_buf.shape[0]
                    datum.width = serialized_pair_buf.shape[1]
                    datum.data = serialized_pair_buf.tostring() # if numpy < 1.9
            
                    # set label            
                    datum.label = int(labels_serial[i])
            
                    # set key_value pair            
                    str_id = '{:08}'.format(i)
                    
                    # The encode is only essential in Python 3
                    txn.put(str_id.encode('ascii'), datum.SerializeToString())
        # done
        
    def cleanup(self):
        print("Running cleanup...")
        self.env.close()
    
    # done
