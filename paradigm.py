#!/usr/bin/env python3 

# Learn paradigm functions

import time, sys, json, argparse
from collections import defaultdict, Counter, namedtuple
from operator import itemgetter
from functools import reduce
from datetime import timedelta

from toolz import pipe

import regex as re
import pandas as pd
import numpy as np

#from keras.models import Sequential, model_from_json
#from keras.layers.core import Dense, RepeatVector
#from keras.layers.merge import Concatenate
#from keras.layers.wrappers import TimeDistributed
#from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar

from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, TimeDistributed, Masking, Bidirectional, Merge
#from keras.layers.merge import Concatenate, Multiply
from keras.layers.recurrent import LSTM
from keras import backend as K


class Paradigms(object):

    def __init__(self, data):

        self.maxlen = max(len(f) for f in data['form'])

        self.charset = sorted(reduce(set.union, map(set, data['form'])))
        self.char_decode = dict(enumerate(self.charset))
        self.char_encode = dict((c, i) for (i, c) in self.char_decode.items())

        self.lexeme = dict((f,i) for (i,f) in enumerate(sorted(pd.unique(data['lexeme']))))
        self.features = dict((f,i+len(self.lexeme)) for (i,f) in enumerate(sorted(pd.unique(data['features']))))

        self.M = len(self.lexeme) + len(self.features)
        self.C = len(self.charset)

    def N(self, data):
        """Number of items to be predicted in a dataset (first char of each form is always given)."""
        return sum(len(f)-1 for f in data['form'])

    def generator(self, data, batch_size):

        x1 = np.zeros((batch_size, self.M), dtype=np.bool)
        x2 = np.zeros((batch_size, self.maxlen, self.C), dtype=np.bool)
        y = np.zeros((batch_size, self.C), dtype=np.bool)
        i = 0

        while True:

            data = data.sample(frac=1)

            for form, lex, feat in data.itertuples(index=False):
                form = np.array([self.char_encode[c] for c in form])

                for j in range(len(form)-1):

                    x1[i, self.lexeme[lex]] = 1
                    x1[i, self.features[feat]] = 1
                    p = self.maxlen-(j+1)
                    #for k,c in enumerate(form[:j+1]):
                    #    x2[i, p+k, c] = 1
                    x2[i, range(p,p+j+1), form[:j+1]] = 1
                    y[i, form[j+1]] = 1

                    i += 1
                
                    if i == batch_size:
                        yield ([x1, x2], y)
                        i = 0
                        x1[:] = 0
                        x2[:] = 0
                        y[:] = 0

            if i > 0:
                yield ([x1[:i], x2[:i]], y[:i])
                i = 0
                x1[:] = 0
                x2[:] = 0
                y[:] = 0
                
    def eval(self, model, testData, return_errors=False, **kwargs):
    
        start = time.time()
        elap = 0.0
        
        batch_size = 20000
        B = 5
        DTYPE = np.bool
        corr = 0
        total = 0
        
        start_char = self.char_encode['<']
        end_char = self.char_encode['>']
        
        if kwargs['verbose']:
            progbar = Progbar(target=len(testData))
        so_far = 0

        for m in range(0, len(testData), batch_size):
        
            batch = testData[m:min(m+batch_size,len(testData))]
            N = len(batch)
            so_far += N
            
            ## Morphosyntactic features

            x1 = np.zeros((N, B, self.M), dtype=DTYPE)
            for i, (form, lex, feat) in enumerate(batch.itertuples(index=False)):
                x1[i,:,self.lexeme[lex]] = 1
                x1[i,:,self.features[feat]] = 1

            ## Initialize beams

            Item = namedtuple('Item', ['score', 'word'])

            beam = [list() for _ in range(N)]
            for j in range(N):
                cand = np.zeros((self.maxlen, self.C), dtype=np.bool)
                cand[-1, start_char] = 1
                beam[j].append(Item(score=1.0, word=cand))
                for _ in range(B-1):
                    beam[j].append(Item(score=-1.0, word=cand))

            x2 = np.zeros((N, B, self.maxlen, self.C), dtype=DTYPE)
            for i in range(self.maxlen-1):
                for j in range(N):
                    for b in range(B):
                        x2[j, b, :, :] = beam[j][b].word
                x1.shape = (N*B, self.M)
                x2.shape = (N*B, self.maxlen, self.C)
                t0 = time.time()
                preds = model.predict([x1, x2], verbose=False, batch_size=1000)
                elap += time.time() - t0
                preds.shape = (N, B, self.C)
                x1.shape = (N, B, self.M)
                x2.shape = (N, B, self.maxlen, self.C)

                new_beam = [list() for _ in range(N)]
                for j in range(N):
                    for b in range(B):
                        u = beam[j][b]
                        if u.word[-1, end_char] == 1:
                            shufflein(new_beam[j], u, B)
                        else:
                            for k in range(self.C):
                                p = u.score * preds[j, b, k]
                                if len(new_beam[j]) < B or p > new_beam[j][-1].score:
                                    t = np.roll(u.word, -1, axis=0)
                                    t[-1, k] = 1
                                    shufflein(new_beam[j], Item(score=p, word=t), B)
                beam = new_beam

            wrong = [ ]    
            for i, (form, lex, feat) in enumerate(batch.itertuples(index=False)):
                word = [self.char_decode[c] for c in np.argmax(beam[i][0].word, axis=1)]
                try:
                    word = word[rindex(word, '<'):word.index('>')+1]
                except ValueError:
                    pass

                total += 1
                if word == form:
                    corr += 1
                else:
                    #form = ''.join(form)
                    #word = ''.join(word)
                    #print(form, lex, feat, word)
                    #for j in range(len(beam[i])):
                    #    print ('**', beam[i][j][0], 
                    #        ''.join(self.char_decode[c] for c in np.argmax(beam[i][j][1], axis=1)))
                    #wrong.append((form, lex, feat, word))
                    pass

            if kwargs['verbose']:
                progbar.update(so_far)
        
        score = corr/total*100.

        print(elap,time.time()-start-elap)
        
        if return_errors:
            return wrong
        else:
            return score

def shufflein(L, x, m):
    L.append(x)
    N = len(L)
    i = N - 2
    while (i >= 0) and (L[i+1].score > L[i].score):
        L[i], L[i+1] = L[i+1], L[i]
        i = i - 1
    if N > m:
        del L[-1]

def rindex(alist, value):
    ## http://stackoverflow.com/questions/9836425/equivelant-to-rindex-for-lists-in-python
    try:
        result = len(alist) - alist[-1::-1].index(value) -1
    except ValueError:
        result = alist
    return result

def baseline(train, test):

    forms = defaultdict(Counter)
    for form, lex, feat in train.itertuples(index=False):
        form = ''.join(form)
        forms[lex][form] += 1        

    correct = 0
    total = 0
    for form, lex, feat in test.itertuples(index=False):
        form = ''.join(form)
        best = forms[lex].most_common(1)
        if best and form == best[0][0]:
            correct += 1
        total += 1

    return correct/total * 100.

class M(Masking):
    def call(self, inputs, mask=None):
        boolean_mask = K.all(K.not_equal(inputs, self.mask_value),
                             axis=-1, keepdims=True)
        return inputs * K.cast(boolean_mask, K.floatx())



def paradigms(data, index, **kwargs):


    t0 = time.time()    

    # Build model

    train = data[~index]
    test = data[index]
    P = Paradigms(data)
    
    print('** Compile model')

    cell_input = Input(shape=(P.M,))
    cell = pipe(cell_input,
                Dense(kwargs['d_dense'], activation='linear'),
                RepeatVector(P.maxlen))

    context_input = Input(shape=(P.maxlen, P.C))
    context = pipe(context_input,
                   TimeDistributed(Dense(kwargs['d_context'], activation='linear')))

    #merged = Concatenate()([cell, context])
    merged = Merge(mode='concat')([cell, context])
    rnn = pipe(merged,
               LSTM(kwargs['d_rnn'], return_sequences=False, unroll=True),
               Dense(P.C, activation='softmax'))

    model = Model(input=[cell_input, context_input], output=[rnn])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    if kwargs['saveModel']:
        print('** Save model to', kwargs['saveModel'])
        open(kwargs['saveModel'], 'w').write(model.to_json())

    # train

    print('** Train model')

    if kwargs['loadWeights']:
        print('** Load weights from', kwargs['loadWeights'])
        model.load_weights(kwargs['loadWeights'])
    else:
        if kwargs['verbose']:
            v = 1
        else:
            v = 2
        fit = model.fit_generator(P.generator(train, batch_size=kwargs['batch_size']),
                                  P.N(train), nb_epoch=kwargs['epochs'], verbose=v, pickle_safe=True)

    if kwargs['saveWeights']:
        print('** Save weights to', kwargs['saveWeights'])
        model.save_weights(kwargs['saveWeights'], overwrite=True)


    # evaluation by beam search

    if len(test) > 0:
        score = P.eval(model, test, **kwargs)
        base = baseline(train, test)
        print('*** Elapsed time:', timedelta(seconds=time.time()-t0))
        return base, score
    else:
        print('*** Elapsed time:', timedelta(seconds=time.time()-t0))
        return None, None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
        description='Model a solution to the Paradigm Cell Filling Problem using a recurrent neural net')
    
    parser.add_argument('--verbose', action='store_true',
                   help='verbose output')

    parser.add_argument('datafile', metavar='datafile', type=str, 
                   help='CSV file to read data from')
    parser.add_argument('--spaces', action='store_true', 
                   help='Segments in input forms separated by spaces')

    data = parser.add_argument_group('data', description='xx')
    data.add_argument('--train', metavar='P', type=float, default=1.0,
                   help='Fraction of data to use for training')
    data.add_argument('--cv', metavar='K', type=int, default=0,
                   help='Perform k-fold cross validation')
    data.add_argument('--saveModel', metavar='FILE', type=str,
                   help='JSON file to store model to')
    data.add_argument('--loadWeights', metavar='FILE', type=str,
                   help='HDF5 file to read weights from')
    data.add_argument('--saveWeights', metavar='FILE', type=str,
                   help='HDF5 file to store weights to')

    model = parser.add_argument_group('model hyperparameters', description='xx')
    model.add_argument('--d_context', metavar='N', type=int, default=8,
                   help='size of context layer')
    model.add_argument('--d_dense', metavar='N', type=int, default=128,
                   help='size of dense layer')
    model.add_argument('--d_rnn', metavar='N', type=int, default=256,
                   help='size of recurrent layers')
    model.add_argument('--n_rnn', metavar='N', type=int, default=1,
                   help='number of recurrent layers')
    model.add_argument('--rnn', metavar='TYPE', choices=['SRN', 'LSTM'], default='LSTM',
                   help='type of recurrent layers (SRN or LSTM)')

    train = parser.add_argument_group('training hyperparameters', description='xx')
    train.add_argument('--epochs', metavar='N', type=int, default=15,
                   help='number of training epochs')
    train.add_argument('--batch_size', metavar='N', type=int, default=128,
                   help='mini-batch size')

    args = vars(parser.parse_args())
    args['tag'] = '>>>>>'  # something to grep for in the logs

    ## read data

    print('** Read data', args['datafile'])

    data = pd.read_csv(args['datafile'], sep='\t', names=['form', 'lexeme', 'features'])
    if args['spaces']:
        data['form'] = [['<'] + f.split(' ') + ['>'] for f in data['form']]
    else:
        data['form'] = [['<'] + re.findall(r'\X', f) + ['>'] for f in data['form']]

    if args['cv']:
        index = np.random.randint(1, args['cv']+1, len(data))
    elif args['train']:
        index = np.array([np.random.random() > float(args['train']) for i in range(len(data))], dtype=np.int)
    else:
        index = np.zeros((len(data)), dtype=np.int)

    for k in range(1, max(index)+1):
        print('** Start run', k)
        print(len(data), sum(index==k))
        args['baseline'], args['score'] = paradigms(data, index==k, **args)
        print(args)
