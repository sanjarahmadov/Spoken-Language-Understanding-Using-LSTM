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

from utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir, count_of_words_and_sentences
from nn_helpers import myMLP, train_nn, Adam

# Otherwise the deepcopy fails
import sys
sys.setrecursionlimit(3000)

class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, de, cs, normal=True, layer_norm = False):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels

        # Implement layer normalization here
        if layer_norm is True:
            
            # Define gain parameters and add it to self.params
            self.g = theano.shared(name='g',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh))
                                .astype(theano.config.floatX))
            
            self.params += [self.g]
            
            def recurrence(x_t, h_tm1):
                a_t = T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh)
                m_t = a_t.sum()/nh
                theta_t = (((a_t - m_t)**2).sum()/nh)**0.5
                h_t = T.nnet.sigmoid((self.g/theta_t)*(a_t-m_t) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
                return [h_t, s_t]
            
        else:
            def recurrence(x_t, h_tm1):
                h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
                return [h_t, s_t]            

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
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
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
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


def test_rnn(**kwargs):
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
        'win': 7,
        'nhidden': 200,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 60,
        'savemodel': False,
        'normal': True,
        'layer_norm': False,
        'minibatch_size':10,
        'moving_avg':3,
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
    #train_set_2, valid_set_2, test_set_2, dic_2 = load_data(2)
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
    
    #groundtruth_valid = [[idx2label[x] for x in y] for y in valid_y]
    #words_valid = [[idx2word[x] for x in w] for w in valid_lex]
    groundtruth_test = [[idx2label[x] for x in y] for y in test_y]
    words_test = [[idx2word[x] for x in w] for w in test_lex]

    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'],
        normal=param['normal'],
        layer_norm=param['layer_norm']
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
                rnn.train(x, y, param['win'], param['clr'])

            predictions_test = [[idx2label[x] for x in rnn.classify(numpy.asarray(
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
                    rnn.save(param['folder'])

                best_rnn = copy.deepcopy(rnn)
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
                rnn = best_rnn

            if param['clr'] < 1e-5:
                break

    print('BEST RESULT: epoch', param['be'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
    
    return rnn, dic