#!/opt/local/bin/python3

### generate tables for Malouf (2017)

from glob import glob
from toolz import concat
import regex as re
import pandas as pd
import numpy as np

## table 1 : data summary

table1 = [ ]
for datafile in glob('data/*.dat.gz'):
    data = pd.read_csv(datafile, sep='\t', names=['form', 'lexeme', 'features', 'lemma'])
    if 'irish' in datafile:
        charset =  set(concat(f.split(' ') for f in data['form']))
    else:
        charset =  set(concat(re.findall(r'\X', f) for f in data['form']))
    maxlen = max(len(f) for f in data['form']) + 1
    row = {'datafile':datafile,
            'n_forms':len(data),
            'n_lexemes': len(pd.unique(data['lexeme'])),
            'n_cells': len(pd.unique(data['features'])),
            'maxlen': maxlen,
            'n_chars': len(charset)}
    table1.append(row)
table1 = pd.DataFrame(sorted(table1, key=lambda d:d['datafile']))
table1 = table1[['datafile', 'n_chars', 'maxlen', 'n_cells', 'n_lexemes', 'n_forms']]
print(table1.to_latex(index=False))

## table 2 : experimental results

table2 = [ ]
for datafile in glob('logs/output.*'):
    for line in open(datafile,'rt'):
        if '>>>>>' in line:
            line = eval(line)
            row = {'datafile':line['datafile'],
                   'baseline':line['baseline'],
                   'accuracy':line['score']}
            table2.append(row)
table2 = pd.DataFrame(sorted(table2, key=lambda d:d['datafile']))
table2 = table2[['datafile','baseline', 'accuracy']]
print(table2.groupby('datafile').agg([np.mean, np.std]).to_latex(float_format='%.2f'))
