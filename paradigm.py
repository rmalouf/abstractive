#!/opt/local/bin/python3

# Learn paradigm functions

import argparse
from collections import defaultdict, Counter, namedtuple
from functools import reduce
from toolz import pipe

import regex as re
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, TimeDistributed, Merge
from keras.layers.recurrent import LSTM
from keras.utils.generic_utils import Progbar
from keras.callbacks import EarlyStopping

from baseline import baseline

class Paradigms(object):

    def __init__(self, data):

        self.maxlen = max(len(f) for f in data['form']) + 1

        self.charset = sorted(reduce(set.union, map(set, data['form']))) + ['<', '>']
        self.char_decode = dict(enumerate(self.charset))
        self.char_encode = dict((c, i) for (i, c) in self.char_decode.items())

        self.lexeme = dict((f,i) for (i,f) in enumerate(sorted(pd.unique(data['lexeme']))))
        self.features = dict((f,i+len(self.lexeme)) for (i,f) in enumerate(sorted(pd.unique(data['features']))))

        self.M = len(self.lexeme) + len(self.features)
        self.C = len(self.charset)

    def N(self, data):
        """Number of items to be predicted in a dataset (add one for end of word marker)"""
        return sum(len(f)+1 for f in data['form'])

    def generator(self, data, batch_size):

        x1 = np.zeros((batch_size, self.M), dtype=np.bool)
        x2 = np.zeros((batch_size, self.maxlen, self.C), dtype=np.bool)
        y = np.zeros((batch_size, self.C), dtype=np.bool)
        i = 0

        while True:
            data = data.sample(frac=1)
            for item in data.itertuples(index=False):
                form = np.array([self.char_encode[c] for c in ['<'] + item.form + ['>']])
                for j in range(len(form)-1):
                    x1[i, self.lexeme[item.lexeme]] = 1
                    x1[i, self.features[item.features]] = 1
                    p = self.maxlen-(j+1)
                    x2[i, range(p, self.maxlen), form[:j+1]] = 1
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
    
        batch_size = 20000
        B = 5
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

            x1 = np.zeros((N, B, self.M), dtype=np.bool)
            for i, item in enumerate(batch.itertuples(index=False)):
                x1[i,:,self.lexeme[item.lexeme]] = 1
                x1[i,:,self.features[item.features]] = 1

            ## Initialize beams

            Item = namedtuple('Item', ['score', 'word'])

            beam = [list() for _ in range(N)]
            for j in range(N):
                cand = np.zeros((self.maxlen, self.C), dtype=np.bool)
                cand[-1, start_char] = 1
                beam[j].append(Item(score=1.0, word=cand))
                for _ in range(B-1):
                    beam[j].append(Item(score=0.0, word=cand))

            x2 = np.zeros((N, B, self.maxlen, self.C), dtype=np.bool)
            for i in range(self.maxlen-1):
                for j in range(N):
                    for b in range(B):
                        x2[j, b, :, :] = beam[j][b].word
                x1.shape = (N*B, self.M)
                x2.shape = (N*B, self.maxlen, self.C)
                preds = model.predict([x1, x2], verbose=False, batch_size=batch_size)
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

            for i, row in enumerate(batch.itertuples(index=False)):
                word = [self.char_decode[c] for c in np.argmax(beam[i][0].word, axis=1)]
                try:
                    word = word[(rindex(word, '<') + 1):word.index('>')]
                except ValueError:
                    pass
                total += 1
                ### print('\t'.join([''.join(word),''.join(row.form),row.lexeme,row.features,str(word==row.form)]))
                if word == row.form:
                    corr += 1

            if kwargs['verbose']:
                progbar.update(so_far)
        
        score = corr/total*100.
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
        raise ValueError
    return result

def x_baseline(train, test):

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

def paradigms(data, index, **kwargs):

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

    merged = Merge(mode='concat')([cell, context])
    rnn = pipe(merged,
               LSTM(kwargs['d_rnn'], return_sequences=False, unroll=True, consume_less='gpu', dropout_U=kwargs['dropout']),
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
                                  P.N(train), nb_epoch=kwargs['epochs'], verbose=v, pickle_safe=True,
                                  callbacks=[EarlyStopping(monitor='loss', patience=5)])

    if kwargs['saveWeights']:
        print('** Save weights to', kwargs['saveWeights'])
        model.save_weights(kwargs['saveWeights'], overwrite=True)

    # evaluation by beam search

    if len(test) > 0:
        score = P.eval(model, test, **kwargs)
        return score
    else:
        return 1.0

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
    data.add_argument('--train', metavar='P', type=float,
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
                   help='size of recurrent layer')
    model.add_argument('--dropout', metavar='P', type=float, default=0.0,
                       help='dropout percentage for recurrent layer')

    train = parser.add_argument_group('training hyperparameters', description='xx')
    train.add_argument('--epochs', metavar='N', type=int, default=15,
                   help='number of training epochs')
    train.add_argument('--batch_size', metavar='N', type=int, default=128,
                   help='mini-batch size')

    args = vars(parser.parse_args())
    args['tag'] = '>>>>>'  # something to grep for in the logs

    ## read data

    print('** Read data', args['datafile'])

    data = pd.read_csv(args['datafile'], sep='\t', names=['form', 'lexeme', 'features', 'lemma'])
    if args['spaces']:
        data['form'] = data['form'].str.split(' ')
        if not data['lemma'].isnull().any():
            data['lemma'] = data['lemma'].str.split(' ')
    else:
        data['form'] = [re.findall(r'\X', f) for f in data['form']]
        if not data['lemma'].isnull().any():
            data['lemma'] = [re.findall(r'\X', f) for f in data['lemma']]

    if args['cv']:
        index = np.random.randint(1, args['cv']+1, len(data))
        for k in range(1, max(index)+1):
            print('** Start run', k)
            args['score'] = paradigms(data, index==k, **args)
            if  not data['lemma'].isnull().any():
                args['baseline'] = baseline(data, index==k, **args)
            else:
                args['baseline'] = 0.0
            print(args)
    elif args['train']:
        index = np.array([np.random.random() > float(args['train']) for i in range(len(data))], dtype=np.bool)
        args['score'] = paradigms(data, index, **args)
        if not data['lemma'].isnull().any():
            args['baseline'] = baseline(data, index, **args)
        else:
            args['baseline'] = 0.0
        print(args)
    else:
        index = np.zeros((len(data)), dtype=np.bool)
        args['score'] = paradigms(data, index, **args)
        args['baseline'] = 1.0
        print(args)



