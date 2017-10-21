# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:07:56 2017

preprocesses images for training and validation

@author: c-morikawa
"""
import cv2
import numpy as np
import sys

class Dataset:
    def __init__(self, img, box, path):
        self.image = img
        self.boundingbox = box
        self.filepath = path

def load_cropped_images(image_dir, image_list, crop_padding, force_square_size):
    print len(image_list)
    result_list = []
    keys = sorted(image_list.keys())
    for i in range(0,len(image_list)):
        if i%10 == 0:
            print '.',
        if i%1000 == 0:
            print i
        # construct full path to the file
        img_filepath = image_dir + '/' + keys[i]
        # print(img_filepath)
        # load it as color image
        orig_bgr = cv2.imread(img_filepath, cv2.IMREAD_COLOR)
        # swap channels - if needed
        orig_rgb = cv2.cvtColor(orig_bgr,cv2.COLOR_BGR2RGB)
        orig_img = orig_bgr
        # load again in grayscale: not opptimal, but faster 
        view = cv2.cvtColor(orig_bgr,cv2.COLOR_BGR2GRAY)
        [row, col] = np.where(view < 250)
        #print row
        #print col
        xmin = max(0, min(col) - crop_padding);
        xmax = min(view.shape[1], max(col) + crop_padding);
        ymin = max(0, min(row) - crop_padding);
        ymax = min(view.shape[0], max(row) + crop_padding);
        
        img_cropped = orig_img[ymin:ymax, xmin:xmax, :]
        
        # TODO: resize larger side to 256, and pad 255 with centering
        target_img = np.zeros((force_square_size,force_square_size,3), 'uint8')
        target_img[:,:,:] = 255
        
        img_width = orig_img.shape[1]
        img_height = orig_img.shape[0]
        
        if img_width > img_height:
            new_width = force_square_size
            new_height = int(force_square_size*float(img_height)/float(img_width))
            resized_image = cv2.resize(img_cropped, (new_width, new_height)) 
            y_offset = (force_square_size - new_height)/2
            target_img[y_offset:y_offset+new_height,:,:] = resized_image[:,:,:]
        else:
            new_width = int(force_square_size*float(img_width)/float(img_height))
            new_height = force_square_size
            resized_image = cv2.resize(img_cropped, (new_width, new_height)) 
            x_offset = (force_square_size - new_width)/2
            target_img[:,x_offset:x_offset+new_width,:] = resized_image[:,:,:]
        
        # add to dataset
        result_list.append(Dataset(target_img, [ymin, xmin, ymax, xmax], img_filepath))
    
    # done
    return result_list

"""                            
images = struct();
pos = 0;

for i_img = 1:length(image_list)
    filename = fullfile(image_dir, image_list{i_img});
    
    orig_img = imread(filename);
    
    if (ndims(orig_img) >=3)
        view = rgb2gray(orig_img);
    else
        % if read as gray scale, convert orig image to rgb
        view = orig_img;
        orig_img = cat(3, view, view, view);
    end
    
    [row, col] = find(view < 250);
    xmin = max(0, min(col) - crop_padding);
    xmax = min(size(view, 2), max(col) + crop_padding);
    ymin = max(0, min(row) - crop_padding);
    ymax = min(size(view, 1), max(row) + crop_padding);

    width = xmax - xmin + 1;
    height = ymax - ymin + 1;
    img_cropped = imcrop(orig_img, [xmin, ymin, width, height]);

    % hos: recompute width, height (sometimes gets cropped out of bound)
    [height, width, dd] = size(img_cropped);
    
    if force_square_size > 0
        if height > width
            img_cropped = imresize(img_cropped, [force_square_size NaN]);
            pre_padding = floor((force_square_size - size(img_cropped, 2)) / 2);
            img_cropped = padarray(img_cropped, [0 pre_padding 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 2);
            img_cropped = padarray(img_cropped, [0 post_padding 0], 255, 'post');
        elseif width > height
            img_cropped = imresize(img_cropped, [NaN force_square_size]);
            pre_padding = floor((force_square_size - size(img_cropped, 1)) / 2);
            img_cropped = padarray(img_cropped, [pre_padding 0 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 1);
            img_cropped = padarray(img_cropped, [post_padding 0 0], 255, 'post');
        else
            img_cropped = imresize(...
                img_cropped, [force_square_size force_square_size]);
        end
        assert(size(img_cropped, 1) == ...
            force_square_size && size(img_cropped, 2) == force_square_size);
    end
    
    pos = pos + 1;
    images(pos).img = img_cropped;
    images(pos).crop_bbox = [ymin, xmin, ymax, xmax];
    images(pos).filename = filename;
    
    fprintf('Loaded %d/%d images(%.1f%%)\n', pos, length(image_list), ...
        100*pos / length(image_list));
end
""" 
