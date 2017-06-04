#!/usr/bin/env python3

## create Maltese verb form database from Apertium translation lexicon
#
#  lt-expand apertium-mlt-ara.mlt.dix | gzip -c > maltese.txt.gz

import re, gzip
import pandas as pd

data = [ ]
with gzip.open('maltese.txt.gz', 'rt') as f:
    for line in f:
        # find verbs, but exclude clitics and RL forms (they're an artefact of Apertium's transfer mechanism)
        if '<vblex>' in line and '+' not in line and ':<:' not in line:
            form, gloss = line.strip().split(':', 1)
            match = re.search(r'(?:>:)?([^<]+)(.*)', gloss)
            lexeme, features = match.group(1), match.group(2)
            data.append({'form':form, 'lexeme':lexeme, 'features':features})

data = pd.DataFrame(data, columns=['form', 'lexeme', 'features'])
data['lemma'] = data['lexeme']
data = data.drop_duplicates(subset=['lexeme','features'], keep='last')
data.to_csv('maltese.dat.gz', compression='gzip', header=None, sep='\t', 
                columns=['form', 'lexeme', 'features', 'lemma'], index=False)
