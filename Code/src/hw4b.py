"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict
import copy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import random

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn, Adam

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(3000)

def gen_parity_pair(nbit, num, for_rnn = False):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    if for_rnn:
        Y = numpy.empty((num, nbit))
        for i in range(X.shape[1]):
            Y[:,i] = numpy.mod(numpy.sum(X[:,:i+1], axis=1), 2)
    else:
        Y = numpy.mod(numpy.sum(X, axis=1), 2)

    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    def __init__(self, n_in, n_hidden, n_out, normal=True):
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, n_hidden))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (n_hidden, n_out))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(n_out,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]
        x = T.fmatrix()
        y = T.ivector()
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]            
        
        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])
        
        p_y_given_x = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x[-1,:])
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')

        nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])
        
        gradients = T.grad(nll, self.params)
        
        updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, gradients))
                                       
        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred)
        self.train = theano.function(inputs=[x, y, lr],
                                              outputs=nll,
                                              updates=updates)
        
#TODO: implement LSTM class to learn parity function
class LSTM(object):
    def __init__(self, n_in, n_hidden, n_out, normal=True):
        # parameters of the model
        self.wi = theano.shared(name='wi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, n_hidden))
                                .astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, n_hidden))
                                .astype(theano.config.floatX))
        self.wc = theano.shared(name='wc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, n_hidden))
                                .astype(theano.config.floatX))
        self.wf = theano.shared(name='wf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, n_hidden))
                                .astype(theano.config.floatX))
        
        self.ui = theano.shared(name='ui',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.uo = theano.shared(name='uo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.uc = theano.shared(name='uc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.uf = theano.shared(name='uf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))    
        
        self.vi = theano.shared(name='vi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.vo = theano.shared(name='vo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))               
        self.vf = theano.shared(name='vf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))  
        
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))        
        self.bc = theano.shared(name='bc',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (n_hidden, n_out))
                               .astype(theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(n_out,
                               dtype=theano.config.floatX))
        
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        
        #self.h0 = T.tanh(self.c0)
        
        # bundle
        self.params = [self.wi, self.wo, self.wc, self.wf, self.ui, self.uo, self.uc, self.uf,
                       self.vi, self.vo, self.vf, self.bi, self.bo, self.bc, self.bf, self.w, self.b, self.c0, self.h0]
        
        x = T.fmatrix()
        y = T.ivector()
        
        def recurrence(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wi) + T.dot(h_tm1, self.ui) + T.dot(c_tm1, self.vi) + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wf) + T.dot(h_tm1, self.uf) + T.dot(c_tm1, self.vf) + self.bf)
            temp = T.tanh(T.dot(x_t, self.wc) + T.dot(h_tm1, self.uc) + self.bc)
            c_t = f_t * c_tm1 + i_t * temp              
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wo) + T.dot(h_tm1, self.uo) + T.dot(c_t, self.vo) + self.bo)
            h_t = o_t * T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, c_t, s_t]            
        
        [h, c, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, self.c0, None],
                                n_steps=x.shape[0])
        
        p_y_given_x = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x[-1,:])
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')

        nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])
        gradients = T.grad(nll, self.params)
        
        updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, gradients))
        
        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred)
        self.train = theano.function(inputs=[x, y, lr],
                                              outputs=nll,
                                              updates=updates)


#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit, n_hidden = 100, n_hiddenLayers = 1, L1_reg = 0.00, L2_reg = 0.00,
                    learning_rate = 0.1, batch_size = 100, n_epochs = 100, verb = True):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: use your MLP and comment out the classifier object above
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_out=2,
        n_hiddenLayers=n_hiddenLayers
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_nn(train_model, validate_model, test_model,
                n_train_batches, n_valid_batches, n_test_batches, n_epochs,
                verbose = verb)
    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    # process input arguments
    param = {
        'n_bit': 8,
        'lr': 0.0970806646812754,
        'verbose': True,
        'decay': True,
        'nhidden': 200,
        'seed': 345,
        'nepochs': 60}
    
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))
    
    n_bit = param['n_bit']
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000, True)
    valid_set = gen_parity_pair(n_bit, 500, True)
    test_set  = gen_parity_pair(n_bit, 100, True)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    print('... building the model')
    rnn = RNN(
        n_in = n_bit,
        n_hidden = param['nhidden'],
        n_out = 2)
    
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    
    # train with early stopping on validation set
    print('... training')
    best_valid = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        
        shuffle([train_set_x.eval(), train_set_y.eval()], param['seed'])
        
        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_set_x.get_value(), train_set_y.eval())):

            rnn.train(x.reshape((x.shape[0], 1)).astype('float32'), y, param['clr'])
            

        # evaluation // back into the real world : idx -> words
        predictions_test = numpy.array([rnn.classify(x.reshape((x.shape[0], 1)).astype('float32')) 
                                        for x in test_set_x.get_value()])
                                        
        predictions_valid =  numpy.array([rnn.classify(x.reshape((x.shape[0], 1)).astype('float32')) 
                                          for x in valid_set_x.get_value()])
        print("Epoch:", e)
        # evaluation // compute the accuracy using conlleval.pl
        res_test = (predictions_test.flatten() == test_set_y.eval()[:,-1]).sum()*100./predictions_test.shape[0]
        res_valid = (predictions_valid.flatten() == valid_set_y.eval()[:,-1]).sum()*100./predictions_valid.shape[0]
        
        if res_valid > best_valid:

            best_rnn = copy.deepcopy(rnn)
            best_valid = res_valid

            if param['verbose']:
                print('NEW BEST: epoch %d valid %.2f best test %.2f' % (e, res_valid, res_test))

            param['bv'], param['bt'] = res_valid, res_test
            param['be'] = e

        if res_test == 100.:
            break
        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch %d valid %.2f best test %.2f' % (param['be'],param['bv'],param['bt']))
    
    return rnn

#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(**kwargs):
    # process input arguments
    param = {
        'n_bit': 8,
        'lr': 0.0970806646812754,
        'verbose': True,
        'decay': True,
        'nhidden': 200,
        'seed': 345,
        'nepochs': 60}
    
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))
    
    n_bit = param['n_bit']
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000, True)
    valid_set = gen_parity_pair(n_bit, 500, True)
    test_set  = gen_parity_pair(n_bit, 100, True)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    print('... building the model')
    lstm = LSTM(
        n_in = n_bit,
        n_hidden = param['nhidden'],
        n_out = 2)
    
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    
    # train with early stopping on validation set
    print('... training')
    best_valid = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        
        shuffle([train_set_x.eval(), train_set_y.eval()], param['seed'])
        
        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_set_x.get_value(), train_set_y.eval())):

            lstm.train(x.reshape((x.shape[0], 1)).astype('float32'), y, param['clr'])
            

        # evaluation // back into the real world : idx -> words
        predictions_test = numpy.array([lstm.classify(x.reshape((x.shape[0], 1)).astype('float32')) 
                                        for x in test_set_x.get_value()])
                                        
        predictions_valid =  numpy.array([lstm.classify(x.reshape((x.shape[0], 1)).astype('float32')) 
                                          for x in valid_set_x.get_value()])
        print("Epoch:", e)
        # evaluation // compute the accuracy using conlleval.pl
        res_test = (predictions_test.flatten() == test_set_y.eval()[:,-1]).sum()*100./predictions_test.shape[0]
        res_valid = (predictions_valid.flatten() == valid_set_y.eval()[:,-1]).sum()*100./predictions_valid.shape[0]
        
        if res_valid > best_valid:

            best_lstm = copy.deepcopy(lstm)
            best_valid = res_valid

            if param['verbose']:
                print('NEW BEST: epoch %d valid %.2f best test %.2f' % (e, res_valid, res_test))

            param['bv'], param['bt'] = res_valid, res_test
            param['be'] = e

        if res_test == 100.:
            break
        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            lstm = best_lstm

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch %d valid %.2f best test %.2f' % (param['be'],param['bv'],param['bt']))
    
    return lstm
    

    
if __name__ == '__main__':
    test_mlp_parity()
