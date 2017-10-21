# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:52:33 2017

@author: c-morikawa
"""

class Dataset:
    def __init__(self, img, box, path):
        self.image = img
        self.boundingbox = box
        self.filepath = path