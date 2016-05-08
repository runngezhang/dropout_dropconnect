"""
This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""


import numpy
import math

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import time
from utils import shared_dataset, load_data
from nn import LogisticRegression, DropoutHiddenLayer, linear_rectifier,DropconnectHiddenLayer, LeNetConvPoolLayer

datasetDictionary = {'MNIST' : [28*28,[28,28],1,10] ,'SVHN':[32*32*3,[32,32],3,10] ,'CIFAR-10':[32*32*3,[32,32],3,10]}

def test_lenet(learning_rate,momentum, n_epochs, nkerns, filter_shapes, pool_sizes,ignore_border,batch_size, verbose,smaller_set, activation,p,dataset,hidden_layer_type,crop,rotate ):
    """
    Wrapper function for testing LeNet on SVHN dataset
        learning_rate - learning rate. Eg - 0.1
        n_epochs - number of epochs
        nkerns - [outputlayer0 kernels,outputlayer1 kernels,outputlayer2_kernels,HiddenLayer neurons]
        filter_shapes - list of filter_shapes for each layer. Eg. [[2,2],[2,2],[3,3]]
        poolsize - list of numbers of poolsize for each layer. Eg. [[1,1,],[1,1],[1,1]]
        ignore_border - specifies if border is to be ignored for the pooling layers. Eg. [False,False,False]
        verbose - prints the output at every epoch. 
        activation - activation function for the CNN and MLP layer. Eg. T.tanh
        p - percentage of elements to be kept. Integer between [0,1] Eg. 0.5 
        hidden_layer_type - type of hidden layer. Either DropoutHiddenLayer or DropconnectHiddenLayer.
        momentum - momentum. Integer between [0,1] Eg. 0.5
        dataset - either 'MNIST' or 'SVHN' or 'CIFAR-10'. type - string. 
        crop - boolean. 
        rotate - boolean. 
    """
    img_size = datasetDictionary[dataset][1]
    channel_size = datasetDictionary[dataset][-2]

    rng = numpy.random.RandomState(23455)

    if smaller_set:
        datasets = load_data(ds_rate=35, theano_shared=True,source = dataset,crop = crop,rotate = rotate)
    else:
        datasets = load_data(ds_rate=None, theano_shared=True,source = dataset,crop = crop,rotate = rotate)


    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]



    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    print 'train_set_x shape :',train_set_x.eval().shape 
    print 'test_set_x shape :',test_set_x.eval().shape 
    print 'valid_set_x shape :',valid_set_x.eval().shape 
    print 'n_train_batches :',n_train_batches
    print 'n_test_batches :',n_test_batches
    print 'n_valid_batches :',n_valid_batches

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

    # Reshape matrix of rasterized images of shape (batch_size, prod(img_size) * channel_size)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, channel_size, img_size[0], img_size[1]))
    
    # Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, channel_size, img_size[0], img_size[1]),
        filter_shape=(nkerns[0], channel_size, filter_shapes[0][0], filter_shapes[0][1]),
        poolsize=pool_sizes[0],
        pool_ignore_border=ignore_border[0]
    )
    
    if ignore_border[0] == True:
        end_size1 = [(img_size[0]-filter_shapes[0][0]+1)//pool_sizes[0][0],
                     (img_size[1]-filter_shapes[0][1]+1)//pool_sizes[0][1]]
    else:
        end_size1 = [int(math.ceil((img_size[0]-filter_shapes[0][0]+1)/pool_sizes[0][0])),
                     int(math.ceil((img_size[1]-filter_shapes[0][1]+1)/pool_sizes[0][1]))]

    print 'layer 0 end : ',end_size1, ' nkerns :',nkerns[0]

    # Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 
                     end_size1[0], end_size1[1]),
        filter_shape=(nkerns[1], nkerns[0], filter_shapes[1][0],
                      filter_shapes[1][1]),
        poolsize=pool_sizes[1],
        pool_ignore_border=ignore_border[1]
    )
    
    if ignore_border[1] == True:
        end_size2 = [(end_size1[0]-filter_shapes[1][0]+1)//pool_sizes[1][0],
                     (end_size1[1]-filter_shapes[1][1]+1)//pool_sizes[1][1]]
    else:
        end_size2 = [int(math.ceil((end_size1[0]-filter_shapes[1][0]+1)/pool_sizes[1][0])), 
                     int(math.ceil((end_size1[1]-filter_shapes[1][1]+1)/pool_sizes[1][1]))]


    print 'layer 1 end : ',end_size2, ' nkerns :',nkerns[1]


    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 
                     end_size2[0], end_size2[1]),
        filter_shape=(nkerns[2], nkerns[1], filter_shapes[2][0],
                      filter_shapes[2][1]),
        poolsize=pool_sizes[2],
        pool_ignore_border=ignore_border[2]
    )
    
    if ignore_border[2] == True:
        end_size3 = [(end_size2[0]-filter_shapes[2][0]+1)//pool_sizes[2][0],
                     (end_size2[1]-filter_shapes[2][1]+1)//pool_sizes[2][1]]
    else:
        end_size3 = [int(math.ceil((end_size2[0]-filter_shapes[2][0]+1)/pool_sizes[2][0])), 
                     int(math.ceil((end_size2[1]-filter_shapes[2][1]+1)/pool_sizes[2][1]))]


    print 'layer 2 end : ',end_size3, ' nkerns :',nkerns[2]


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer3_input = layer2.output.flatten(2)


    print 'layer 2 begin : ',numpy.prod(end_size3)*nkerns[2]

    # construct a fully-connected sigmoidal layer
    layer3 = hidden_layer_type(
        rng=rng,
        input=layer3_input,
        n_in=numpy.prod(end_size3)*nkerns[2],
        n_out=nkerns[3],
        batch_size= batch_size,
        activation=activation,
        p = p,
        is_train = is_train
    )

    print 'layer 2 end : ',nkerns[3]

    consider_constant = [] 
    if hasattr(layer3, 'consider_constant'):
        print 'Found.'
        consider_constant.extend(layer3.consider_constant)

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in=nkerns[3],
         n_out=datasetDictionary[dataset][-1]
    )

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)



    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train : numpy.cast['int32'](0)

        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train : numpy.cast['int32'](0)

        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params


    # Reference for momentum http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert momentum >= 0. and momentum < 1.
    
    momentum =theano.shared(numpy.cast[theano.config.floatX](momentum), name='momentum') 
    updates = []

    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))  
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param,consider_constant=consider_constant)))          



    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train : numpy.cast['int32'](1)

        }
    )


    print('Model build.')

    ##############
    ###TRAIN MODEL #
    ###############
    print('... training')

    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1        
        print "momentum: ", momentum.get_value()    
        print "learning rate: ", learning_rate 
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break

        # adaption of momentum
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # adaption of learning rate    
        learning_rate = learning_rate * 0.985
                

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))



    
# test_lenet(learning_rate= 0.1, n_epochs=200, nkerns=[2,5,4,150], filter_shapes=[[2,2],[2,2],[2,2]], pool_sizes=[[1,1,],[1,1],[1,1]],ignore_border=[False,False,False],batch_size = 20, verbose = True, smaller_set= True, activation =T.tanh, p= 0.5, dataset= 'MNIST', hidden_layer_type = DropconnectHiddenLayer, momentum = 0.5,crop = False,rotate = False)

