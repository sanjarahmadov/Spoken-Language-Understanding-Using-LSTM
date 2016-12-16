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
import theano.typed_list
import random

from utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir, count_of_words_and_sentences
from nn_helpers import myMLP, train_nn, Adam, drop

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(6000)

class BLSTM_ATT(object):
    def __init__(self, n_hidden, n_hidden2, n_out, n_emb, dim_emb, cwind_size, normal):
        """Initialize the parameters for the LSTM

        :type n_hidden: int
        :param nh: dimension of the hidden layer
        
        :type n_hidden2: int
        :param nh: dimension of the second hidden layer if experiemnt is deep
        
        :type n_out: int
        :param nc: number of classes

        :type n_emb: int
        :param ne: number of word embeddings in the vocabulary

        :type dim_emb: int
        :param de: dimension of the word embeddings

        :type cwind_size: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.
        
        :type layer_normal: boolean
        :param normal: normalize layer.       

        :type n_hidden2: string
        :param nh: experiment type

        """
        
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (n_emb+1, dim_emb))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))     
        self.W_xi = theano.shared(name='W_xi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (dim_emb * cwind_size, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xo = theano.shared(name='W_xo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (dim_emb * cwind_size, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xc = theano.shared(name='W_xc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (dim_emb * cwind_size, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xf = theano.shared(name='W_xf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (dim_emb * cwind_size, n_hidden))
                                .astype(theano.config.floatX))
        
        self.W_hi = theano.shared(name='W_hi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.W_ho = theano.shared(name='W_ho',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.W_hc = theano.shared(name='W_hc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.W_hf = theano.shared(name='W_hf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))    
        # Diagonal weights
        self.W_ci = theano.shared(name='W_ci',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
                                .astype(theano.config.floatX))
        self.W_co = theano.shared(name='W_co',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
                                .astype(theano.config.floatX))            
        self.W_cf = theano.shared(name='W_cf',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
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
            
        
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        
        self.W_att = theano.shared(name='W_att',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))   
        
        self.W_att2 = theano.shared(name='W_att',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX)) 
        
        # bundle
        self.params = [self.emb, self.W_xi, self.W_xo, self.W_xc, self.W_xf, self.W_hi, self.W_ho, self.W_hc, self.W_hf,
                       self.W_ci, self.W_co, self.W_cf, self.bi, self.bo, self.bc, self.bf, self.c0, self.h0]
        
        self.W_xi2 = theano.shared(name='W_xi2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xo2 = theano.shared(name='W_xo2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xc2 = theano.shared(name='W_xc2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.W_xf2 = theano.shared(name='W_xf2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        
        self.W_hi2 = theano.shared(name='W_hi2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))
        self.W_ho2 = theano.shared(name='W_ho2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.W_hc2 = theano.shared(name='W_hc2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))        
        self.W_hf2 = theano.shared(name='W_hf2',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))
                                .astype(theano.config.floatX))    
        # Diagonal weights
        self.W_ci2 = theano.shared(name='W_ci2',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
                                .astype(theano.config.floatX))
        self.W_co2 = theano.shared(name='W_co2',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
                                .astype(theano.config.floatX))            
        self.W_cf2 = theano.shared(name='W_cf2',
                                value=numpy.diag(numpy.diag(0.2 * numpy.random.uniform(-1.0, 1.0,
                                (n_hidden, n_hidden))))
                                .astype(theano.config.floatX))
        
        self.bi2 = theano.shared(name='bi2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.bo2 = theano.shared(name='bo2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))        
        self.bc2 = theano.shared(name='bc2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.bf2 = theano.shared(name='bf2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))        
        
        self.h2 = theano.shared(name='h2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        self.c2 = theano.shared(name='c2',
                                value=numpy.zeros(n_hidden,
                                dtype=theano.config.floatX))
        
        self.params += [self.W_xi2, self.W_xo2, self.W_xc2, self.W_xf2, self.W_hi2, self.W_ho2, self.W_hc2, self.W_hf2,
                       self.W_ci2, self.W_co2, self.W_cf2, self.bi2, self.bo2, self.bc2, self.bf2, self.c2, self.h2]     
        
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], dim_emb * cwind_size))
        y_sentence = T.ivector('y_sentence')  # labels

        self.w2 = theano.shared(name='w2',
                           value=0.2 * numpy.random.uniform(-1.0, 1.0,
                           (n_hidden, n_out))
                           .astype(theano.config.floatX))

        self.b2 = theano.shared(name='b2',
                           value=numpy.zeros(n_out,
                           dtype=theano.config.floatX)) 

        self.params += [self.w2, self.b2, self.W_att, self.W_att2]
        #"""
        def encoder_recurrence(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.bf)
            temp = T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.bc)
            c_t = f_t * c_tm1 + i_t * temp       
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co) + self.bo)
            h_t = o_t * T.tanh(c_t)
            
            h_t = drop(h_t, 0.7)
            c_t = drop(c_t, 0.7)
            
            alpha = T.nnet.softmax(T.dot(T.tanh(h_t), self.W_att) + T.dot(T.tanh(c_t), self.W_att2))
            r = T.tanh((alpha*h_t).sum(axis=0))
            c_t = r
                   
            return [h_t, c_t]     

        [h_1, c_1], _ = theano.scan(fn=encoder_recurrence,
                                sequences=x,
                                outputs_info=[self.h0, self.c0],
                                n_steps=x.shape[0])
        
        [h_2, c_2], _ = theano.scan(fn=encoder_recurrence,
                                sequences=x[::-1],
                                outputs_info=[self.h0, self.c0],
                                n_steps=x.shape[0])      
        h = h_1 + h_2
        
        def decoder_recurrence(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi2) + T.dot(h_tm1, self.W_hi2) + T.dot(c_tm1, self.W_ci2) + self.bi2)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf2) + T.dot(h_tm1, self.W_hf2) + T.dot(c_tm1, self.W_cf2) + self.bf2)
            temp = T.tanh(T.dot(x_t, self.W_xc2) + T.dot(h_tm1, self.W_hc2) + self.bc2)
            c_t = f_t * c_tm1 + i_t * temp       
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo2) + T.dot(h_tm1, self.W_ho2) + T.dot(c_t, self.W_co2) + self.bo2)
            h_t = o_t * T.tanh(c_t)        
            
            s_t = T.nnet.softmax(T.dot(h_t, self.w2) + self.b2)
            return [h_t, c_t, s_t]     

        [h, c, s], _ = theano.scan(fn=decoder_recurrence,
                                sequences=h,
                                outputs_info=[self.h2, self.c2, None],
                                n_steps=x.shape[0])   
        
        
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)  
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        
        sentence_gradients = T.grad(sentence_nll, self.params)
        
        
        sentence_updates = OrderedDict(Adam(self.params, sentence_gradients, lr/100))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        
        self.see_res = theano.function(inputs=[idxs], outputs=[s.shape], on_unused_input='ignore')
        
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = [numpy.asarray(x).astype('int32') for x in cwords]
        labels = y
        
        self.sentence_train(words, labels, learning_rate)
        
        if self.normal:
            self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))

def test_blstm_att(**kwargs):
    """
    Wrapper function for training and testing LSTM

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient).

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    # process input arguments
    param = {
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 3,
        'nhidden': 300,
        'nhidden2':300,
        'seed': 345,
        'emb_dimension': 90,
        'nepochs': 40,
        'savemodel': False,
        'normal': True,
        'minibatch_size':4978,
        'folder':'../result'}
    
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')
    train_set, valid_set, test_set, dic = load_data(3)
    
    train_set = list(train_set)
    valid_set = list(valid_set)

    # Add validation set to train set
    for i in range(3):
        train_set[i] += valid_set[i]
    
    # create mapping from index to label, and index to word
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())    
    
    # unpack dataset
    train_lex, train_ne, train_y = train_set
    test_lex, test_ne, test_y = test_set
    
    n_trainbatches = len(train_lex)//param['minibatch_size']

    print("Sentences in train: %d, Words in train: %d" % (count_of_words_and_sentences(train_lex)))
    print("Sentences in test: %d, Words in test: %d" % (count_of_words_and_sentences(test_lex)))
    
    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)
    
    groundtruth_test = [[idx2label[x] for x in y] for y in test_y]
    words_test = [[idx2word[x] for x in w] for w in test_lex]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    
    print('... building the model')
    lstm = BLSTM_ATT(
        n_hidden=param['nhidden'],
        n_hidden2=param['nhidden2'],
        n_out=nclasses,
        n_emb=vocsize,
        dim_emb=param['emb_dimension'],
        cwind_size=param['win'],
        normal=param['normal']
    )
    
    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    
    for e in range(param['nepochs']):

        # shuffle
        shuffle([train_lex, train_ne, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()
        
        for minibatch_index in range(n_trainbatches):
            
        
            for i in range(minibatch_index*param['minibatch_size'],(1 + minibatch_index)*param['minibatch_size']):
                x = train_lex[i]
                y = train_y[i]
                lstm.train(x, y, param['win'], param['clr'])
        
            predictions_test = [[idx2label[x] for x in lstm.classify(numpy.asarray(
                        contextwin(x, param['win'])).astype('int32'))]
                         for x in test_lex]
            

            # evaluation // compute the accuracy using conlleval.pl
            res_test = conlleval(predictions_test,
                                 groundtruth_test,
                                 words_test,
                                 param['folder'] + '/current.test.txt',
                                 param['folder'])


            if res_test['f1'] > best_f1:

                if param['savemodel']:
                    lstm.save(param['folder'])

                best_lstm = copy.deepcopy(lstm)
                best_f1 = res_test['f1']

                if param['verbose']:
                    print('NEW BEST: epoch %d, minibatch %d/%d, best test F1: %.3f' 
                          %(e, minibatch_index+1, n_trainbatches, res_test['f1']))

                param['tf1'] = res_test['f1']
                param['tp'] = res_test['p']
                param['tr'] = res_test['r']
                param['be'] = e

                os.rename(param['folder'] + '/current.test.txt',
                          param['folder'] + '/best.test.txt')
            else:
                if param['verbose']:
                    print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            print("Decay happened. New Learning Rate:", param['clr'])
            lstm = best_lstm

        if param['clr'] < 0.00001:
            break

    print('BEST RESULT: epoch', param['be'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
    
    return lstm, dic