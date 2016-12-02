"""
Source Code for Homework 4 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/rnnslu.html
"""

from collections import OrderedDict
from itertools import product

import gzip
import numpy
import os
import pickle
import random
import scipy.io
import stat
import subprocess
import sys

import theano
import theano.tensor as T

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def shuffle(lol, seed):
    """
    Suffle inplace each list in the same order

    :type lol: list
    :param lol: list of list as input

    :type seed: int
    :param sedd: seed the shuffling

    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def contextwin(l, win):
    """
    Return a list of list of indexes corresponding to context windows
    surrounding each word in the sentence

    :type win: int
    :param win: the size of the window given a list of indexes composing a sentence

    :type l: list or numpy.array
    :param l: array containing the word indexes

    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def conlleval(p, g, w, filename, script_path):
    """
    Evaluate the accuracy using conlleval.pl

    :type p: list
    :param p: predictions

    :type g: list
    :param g: groundtruth

    :type w: list
    :param w: corresponding words

    :type filename: string
    :param filename: name of the file where the predictions are written. It
    will be the input of conlleval.pl script for computing the performance in
    terms of precision recall and f1 score.

    """
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    """
    Run conlleval.pl perl script to obtain precision/recall and F1 score.

    :type filename: string
    :param filename: path to the file

    """
    _conlleval = os.path.join('./', 'conlleval.pl')
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        print('Downloading conlleval.pl from %s' % url)
        urllib.urlretrieve(url, _conlleval)
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()).encode('utf-8'))
    stdout = stdout.decode('utf-8')
    out = None

    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    # To help debug
    if out is None:
        print(stdout.split('\n'))
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}

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

def load_data(foldnum=3):
    '''
    Load the ATIS dataset

    :type foldnum: int
    :param foldnum: fold number of the ATIS dataset, ranging from 0 to 4.

    '''

    atis_url = 'http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'
    filename = 'atis.fold'+str(foldnum)+'.pkl.gz'

    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (atis_url + dataset)
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    filename = check_dataset(filename)
    f = gzip.open(filename, 'rb')
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set, dicts = pickle.load(f)
    return train_set, valid_set, test_set, dicts

def grid_search(func, gridfunc=None, verbose=False, **kwargs):
    """
    Generic function for grid search

    :type func: callable
    :param func: function to perform grid search upon.

    :type gridfunc: callable
    :param gridfunc: a function for setting keyword's value based on the
    current parameters. Use this function to synthesize model name, ex.
    "mlp_lr0.001_nhidden100", or to set result directory, ex.
    "./result/test/mlp_hidden100".

    :type verbose: boolean
    :param verbose: to print out grid summaryy or not to

    :type kwargs: dict
    :param kwargs: dictionay of parameters for func. Keys should be valid input
    arguments of func, and their corresponding value could be a scalar or a list
    or values that would iterated over, ex, {'n_epochs:10, n_hidden:[100,150]'}

    Examples:

    >>> def test_func(a=1,b=2,c=3, prod=False):
            if prod is True:
                print("test_func: a*b*c=%d" % (a*b*c,))
            else:
                print("test_func: a=%d, b=%d, c=%d" % (a,b,c))

    >>> grid_search(test_func, verbose=True, **{'a':[1,2,3],'b':1})

    Use wrapper function to set input arguments:

    >>> def test_func_wrap(**kwargs):
            return test_func(b=1, prod=True, **kwargs)
    >>> grid_search(test_func_wrap, verbose=True, **{'a':[1,2,3]})

    Use gridfunc to change data directory

    >>> def gfunc(args):
            return {'dirname':'mlp_'+str(args['n_hidden'])}

    >>> grid_search(test_mlp, gridfunc=gfunc, **{'n_hidden':[100,200,300]})

    """

    # create parameters grid
    makelist = lambda x: x if hasattr(x, '__iter__') else (x,)
    kwargs = OrderedDict({k:makelist(v) for k,v in kwargs.items()})
    grid = product(*kwargs.values())
    n_grid = numpy.prod(map(len,(kwargs.values())))

    print('... starting grid search with %d combinations' % n_grid)
    for i, params in enumerate(grid):
        args = {k:v for k,v in zip(kwargs.keys(),params)}
        if gridfunc is not None:
            args.update(gridfunc(args))
        # print out parameters
        if verbose:
            print('=== Parameter set: %.4d/%.4d, %2.2f%% ===' %
                  (i+1,n_grid,100.*float(i+1)/float(n_grid)))
            for k,v in args.items():
                print('[%s]: %s' % (str(k),str(v)))
            print('... running function %s' % func.__name__)
        func(**args)
    print('... end of grid search\n')

