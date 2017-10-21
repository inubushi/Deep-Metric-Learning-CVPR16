# Deep Metric Learning via Lifted Structured Feature Embedding - Inubushi's Python Fork
I added Python code to the original repository for the benefit of those of us who don't have access to Matlab. There are places where the original code is not optimal, but I did not try to change much because that might make the code difficult to understand.

This repository has the source code and the Stanford Online Products dataset for the paper "Deep Metric Learning via Lifted Structured Feature Embedding" (CVPR16). The paper is available on [cv-foundation](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf). If you just need the Caffe code, check out the [Submodule](https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16). For the loss layer implementation, look at [here](https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp).

## Citing this work (original citation)
If you find this work useful in your research, please consider citing:

    @inproceedings{songCVPR16,
        Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
        Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
        Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

## Installation
1. Install prerequsites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Compile the `Caffe-Deep-Metric-Learning-CVPR16` Github submodule.

## Prerequisites
1. Download pretrained GoogLeNet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
2. Download the ILSVRC12 ImageNet mean file for mean subtraction. Refer to Caffe the ImageNet examples  [here](https://github.com/BVLC/caffe/tree/master/examples/imagenet).
3. Modify and run `code/gen_splits.m` to create train/test split.
4. Modify and run `code/gen_images.m` to prepare the preprocessed images.

## Training Procedure
1. Generate the LMDB file to convert the training set of images to the DB format. Example scripts are in `code/` directory.
 * Modify and run `code/compile.m` to mex compile the cpp files used for LMDB generation.
 * Modify `code/config.m` to set save paths.
 * Run `code/gen_caffe_dataset_multilabel_m128.m` to start the LMDB generation process.
2. Create the `model/train*.prototxt` and `model/solver*.prototxt` files. Please refer to the included `*.prototxt` files in `model/` directory for examples. You also need to provide the path to the ImageNet mean file (usually called `imagenet_mean.binaryproto`) you downloaded in step 2.
3. Inside the caffe submodule, launch the Caffe training procedure.
`caffe/build/tools/caffe train -solver [path-to-training-prototxt-file] -weights [path-to-pretrained-googlenet] -gpu [gpuid]`

## Feature extraction after training
1. Modify and run `code/gen_caffe_validation_imageset.m` to convert the test images to LMDB format.
1. Modify the test set path in `model/extract_googlenet*.prototxt`.
2. Modify the model and test set path and run `code/compute_googlenet_distance_matrix_cuda_embeddings_liftedstructsim_softmax_pair_m128.py`.

## Clustering and Retrieval evaluation code
1. Use `code/evaluation/evaluate_clustering.m` to evaluate the clustering performance.
2. Use `code/evaluation/evaluate_recall.m` to evaluate recall@K for image retrieval.

## Stanford Online Products dataset
You can download the Stanford Online Products dataset (2.9G) from ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
* We also have the text meta data for each product images. Please let us know if you're interested in using them.

## Our Pre-trained Models
You can download our pre-trained models on the Cars196 dataset, the CUB200 dataset and the Online Products dataset (265M) from ftp://cs.stanford.edu/cs/cvgl/pretrained_models.zip

## Licence
MIT Licence
