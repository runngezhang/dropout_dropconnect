
import numpy
import theano
import theano.tensor as T
import copy
import time
from utils import load_data
from nn import LogisticRegression,DropoutHiddenLayer, myMLP ,linear_rectifier,DropconnectHiddenLayer
from theano.tensor.shared_randomstreams import RandomStreams

datasetDictionary = {'MNIST' : [28*28,1,10] ,'SVHN':[32*32*3,3,10] }

def test_mlp(learning_rate, momentum, L1_reg, L2_reg, n_epochs,batch_size, n_hidden, n_hiddenLayers,hidden_layer_type,filename,p = 0.5, dataset = 'MNIST', verbose = True, smaller_set = True,activation = T.tanh):

    '''
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type momentum: float
    :param momentum: momentum for the update. 

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type hidden_layer_type: object
    :param hidden_layer_type : type of hidden layer. 

    :type p : float
    :param p : percentage of the parameters to be kept.

    :type filename : string
    :param filename : Name of the output file which is stored. 

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    '''

    if smaller_set:
        datasets = load_data(ds_rate=35, theano_shared=True,source = dataset)
    else:
        datasets = load_data(ds_rate=None, theano_shared=True,source = dataset)


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
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
    	rng = rng,
    	input = x,
    	n_in = datasetDictionary[dataset][0],
    	n_hidden = n_hidden,
    	n_out = datasetDictionary[dataset][-1],
    	n_hiddenLayers = n_hiddenLayers,
    	hidden_layer_type= hidden_layer_type,
    	p= p,
    	is_train = is_train,
        activation = activation,
        batch_size = batch_size
        )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    consider_constant = []
 
    if hasattr(classifier, 'consider_constant'):
        consider_constant.extend(classifier.consider_constant) 

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y),cost,classifier.output],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
            is_train : numpy.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
            is_train:numpy.cast['int32'](0)
        }
    )



    # Reference for momentum http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
    assert momentum >= 0. and momentum < 1.
    
    momentum =theano.shared(numpy.cast[theano.config.floatX](momentum), name='momentum') 
    updates = []

    for param in  classifier.params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))  
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param, consider_constant=consider_constant)))          
    
    

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train : numpy.cast['int32'](1)
        }
    )    

    print 'Model Build'

    ######################
    ## TRAINING MODEL ####
    ######################

    print '... training'

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
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

    train_cost = []
    test_cost = []
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1        
        print "momentum: ", momentum.get_value()    
        print "learning rate: ", learning_rate 
        cost_for_batch = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            cost_for_batch.append(minibatch_avg_cost)


            # iteration number
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
                    test_losses = []
                    test_cost_for_batch = []
                    test_predictions = []
                    for i in xrange(n_test_batches):
                        loss,cost,ypred = test_model(i)
                        # print ypred
                        test_predictions.extend(ypred)
                        test_losses.append(loss)
                        test_cost_for_batch.append(cost)

                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if patience <= iter:
                done_looping = True
                break
        train_cost.append(numpy.mean(cost_for_batch))
        test_cost.append(numpy.mean(test_cost_for_batch))

        # adaption of momentum
        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * 0.98
            momentum.set_value(numpy.cast[theano.config.floatX](new_momentum))
        # adaption of learning rate    
        learning_rate = learning_rate * 0.985
        # learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))
                

    end_time = time.clock()


    #### SAVES THE OUTPUT in a CSV file. 
    import csv
    outputFilename = open(filename+'.csv','wb')
    writer  = csv.writer(outputFilename,dialect = 'excel')

    for entry in test_predictions:
        writer.writerow([entry])

    writer.writerow([test_score * 100.])

    writer.writerow(train_cost)
    writer.writerow(test_cost)
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

#test_mlp(learning_rate = 0.1, momentum = 0.5, L1_reg= 0.00, L2_reg= 0.0001, n_epochs=20, batch_size=20, n_hidden=5, n_hiddenLayers=2,hidden_layer_type= DropoutHiddenLayer,p = 0.5, dataset = 'MNIST', verbose = True, smaller_set = True,filename = 'mlp_dropconnect_mnist_tanh',activation = T.tanh)










