"""
Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
"""
import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T
import cPickle, gzip, bz2
import copy
from numpy import linspace

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def random_crop_image(image, crop_size, seed):
    ''' given the image and final size after random cropping, returns the cropped image

    :type image: numpy.array (2D)
    :the image to be cropped

    :type crop_size: tuple or list
    :the size of the image after cropping

    '''

    img_size = image.shape
    
    u_end = img_size[0] - crop_size[0]
    l_end = img_size[1] - crop_size[1]
    
    rng = numpy.random.RandomState(seed)
    top = rng.random_integers(0, u_end)
    left = rng.random_integers(0, l_end)

    img_cropped = image[top:(top + crop_size[0]), left:(left + crop_size[1])]
    
    return img_cropped


def training_data_shuffle(img_matrix):
    ''' Given 2D matrix of data, where rows are individual examples, shuffles the order
    of rows
    '''

    numpy.random.shuffle(img_matrix)

def flip_img_horiz(img_matrix):
    ''' Given 2D image matrix, flips the image horizontally

    :type img_matrix: numpy.array
    :image matrix values

    '''
    return numpy.fliplr(img_matrix)

def image_rescale_down(img_matrix, final_size):
    ''' Given 2D image matrix, 'rescales' by subsampling
    '''
    
    img_shape = img_matrix.shape
    assert(final_size[0] < img_shape[0] & final_size[1] < img_shape[1])
    
    range1 = linspace(0, img_shape[0], final_size[0]).astype(int).tolist()
    range2 = linspace(0, img_shape[1], final_size[1]).astype(int).tolist()
    
    image2 = img_matrix[range1]
    
    return image2[:, range2]
    


def image_rotate(img_matrix, seed):
    ''' Given 2D image matrix, rotates 90 deg counterclockwise random # of times
    '''
    
    return numpy.rot90(img_matrix, k = seed)


def subtract_per_pixel_mean(img_matrix):
    ''' Given 2D matrix with each row a flattened image, subtracts the per pixel mean
    from all the rows

    :type img_matrix: numpy.array
    :image matrix values
    
    '''

    row_mean = numpy.mean(img_matrix, 0)
    img_matrix = img_matrix - row_mean

    return img_matrix


def preprocess_data(dataset, source,crop=False,rotate=False,scaling= False):

    rng = numpy.random.RandomState(99)
    
    data_shape = dataset.shape
    
    if source == 'MNIST':
        #size normalized to 20 x 20
        dataset = dataset.reshape(data_shape[0], 28, 28)
        final = dataset        
        if crop == True:
            final = numpy.empty((data_shape[0], 24, 24))

        for i in range(data_shape[0]):
            seed = rng.random_integers(0, 100)
            if crop == True:

                final[i,:,:] = random_crop_image(dataset[i,:,:],
                                                  (24, 24), seed)
            if rotate == True:
                final[i,:,:] = image_rotate(final[i,:,:],seed)

                # final[i,:,:] = image_rescale_down(dataset[i,:,:], (20, 20))

        if crop == True:
            final = final.reshape(data_shape[0], 24*24)
        else:
            final = final.reshape(data_shape[0],28*28)
        
    elif source == 'CIFAR-10':

        #reshape to (n_examples, 3, 32, 32)
        dataset = dataset.reshape(data_shape[0], 3, 32, 32)

        #randomly crop to 24 x 24, randomly do horizontal flip
        final = numpy.empty((data_shape[0], 3, 24, 24))
        
        for i in range(data_shape):
            for j in range(3):
                seed = rng.random_integers(0, 100)
                final[i,j,:,:] = random_crop_image(dataset[i,j,:,:],
                                                  (24, 24), seed)
                seed = rng.random_integers(0, 1)
                if seed == 0:
                    final[i,j,:,:] = flip_img_horiz(final[i,j,:,:])

        final = final.reshape(data_shape[0], 3*24*24)
        
    elif source == 'SVHN':
        
        #reshape to (n_examples, 3, 32, 32)
        dataset = dataset.reshape(data_shape[0], 3, 32, 32)

        #randomly crop to 28 x 28, rotate, 'scale' by 85%
        final = numpy.empty((data_shape[0], 3, 28, 28))
        
        for i in range(data_shape):
            for j in range(3):
                seed = rng.random_integers(0, 100)
                final[i,j,:,:] = random_crop_image(dataset[i,j,:,:],
                                                  (28, 28), seed)
                seed = rng.random_integers(1, 4)
                final[i,j,:,:] = image_rotate(final[i,j,:,:], seed)
                
                final[i,j,:,:] = image_rescale_down(final[i,j,:,:], (24, 24))

        final = final.reshape(data_shape[0], 3*28*28)
        
    elif source == 'NORB':
        #downsample from 108 x 108 to 48 x 48 (FXN NOT DONE YET), rotate, scale (FXN NOT DONE YET)
        pass

    return final

def load_data(ds_rate, theano_shared, source,crop = False, rotate = False,scaling = False):
    ''' Loads the dataset according to the source

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    if source == 'SVHN':
        # copied from implementation in previous homework assignment
        train_set = scipy.io.loadmat('data/train_32x32_SVHN.mat')
        test_set = scipy.io.loadmat('data/test_32x32_SVHN.mat')

        # Convert data format from (3, 32, 32, n_samples) to (n_samples, 3*32*32)
        # Also normalizes data matrix from range [1, 255] to [0, 1] by dividing by 255
        # SVHN data is 3-channels (R, G, B)
        def convert_data_format(data):
            X = data['X'].transpose(2,0,1,3)
            X = numpy.reshape(data['X'],
                          (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                          order='C').T / 255.
            y = data['y'].flatten()
            y[y == 10] = 0
            return (X,y)
        
        train_set = convert_data_format(train_set)
        test_set = convert_data_format(test_set)
        



        # Downsample the training dataset if specified
        train_set_len = len(train_set[1])
        
        if ds_rate is not None:
            train_set_len = int(train_set_len // ds_rate)
            train_set = [x[:train_set_len] for x in train_set]


        # Extract validation dataset from train dataset
        valid_set = [x[-(train_set_len//10):] for x in train_set]
        train_set = [x[:-(train_set_len//10)] for x in train_set]
        
        
    elif source == 'MNIST':
        # see http://deeplearning.net/tutorial/gettingstarted.html for details
        # 1 black/white channel only, each image is 1D ndarray w/28x28 values in [0,1]
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f) 

        train_set = list(train_set)
        valid_set = list(valid_set)
        test_set = list(test_set)

        # Downsample the training dataset if specified
        train_set_len = len(train_set[1])
        
        if ds_rate is not None:
            train_set_len = int(train_set_len // ds_rate)
            train_set = [x[:train_set_len] for x in train_set]

        print 'crop :',crop,'rotate :',rotate
        train_set[0] = preprocess_data(train_set[0],'MNIST',crop= crop,rotate= rotate,scaling = scaling)
        valid_set[0] = preprocess_data(valid_set[0],'MNIST',crop= crop,rotate= rotate,scaling = scaling)
        test_set[0] = preprocess_data(test_set[0],'MNIST',crop= crop,rotate= rotate,scaling = scaling)
            
    elif source == 'CIFAR-10':
        # see https://www.cs.toronto.edu/~kriz/cifar.html for details
        # images stored as R,R,...,R, G,G,...,G, B,B,...,B
        filenames = ['data/data_batch_1', 'data/data_batch_2', 'data/data_batch_3',
                     'data/data_batch_4', 'data/data_batch_5', 'data/test_batch']
        
        i = 0
        for item in filenames:
            i += 1
            fo = open(item, 'rb')
            dict = cPickle.load(fo)
            fo.close()

            # normalize to [0,1] range
            data_array = dict['data']/255.
            label_array = dict['labels']
            
            if i == 1:
                train_set = [data_array, label_array]
            elif i <= 5:
                train_set[0] = numpy.concatenate((train_set[0], data_array))
                train_set[1] = numpy.concatenate((train_set[1], label_array))
            else:
                test_set = [data_array, label_array]

        # Downsample the training dataset if specified
        train_set_len = len(train_set[1])
        
        if ds_rate is not None:
            train_set_len = int(train_set_len // ds_rate)
            train_set = [x[:train_set_len] for x in train_set]

        # Extract validation dataset from train dataset
        valid_set = [x[-(train_set_len//10):] for x in train_set]
        train_set = [x[:-(train_set_len//10)] for x in train_set]

        
    elif source == 'NORB':
        # see http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/ for details
        # labels are in {0, 1, 2, 3, 4}

        train_set_x = scipy.io.loadmat('data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
        test_set_x = scipy.io.loadmat('data/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
        train_set_y = scipy.io.loadmat('data/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
        test_set_y = scipy.io.loadmat('data/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')


        
    else:
        print('Invalid dataset!')
        exit()        
        
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval
