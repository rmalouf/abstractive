#!/usr/bin/env python3

# The French verb paradigms are from Flexique

# http://www.llf.cnrs.fr/sites/sandbox.linguist.univ-paris-diderot.fr/files/statiques/flexique/flexique.zip

import requests, zipfile, io
import pandas as pd

URL = 'http://www.llf.cnrs.fr/sites/sandbox.linguist.univ-paris-diderot.fr/files/statiques/flexique/flexique.zip'

r = requests.get(URL)
z = zipfile.ZipFile(io.BytesIO(r.content))

data = pd.read_csv(io.BytesIO(z.read('data/vlexique.csv')), header=None, na_values=['#DEF#', '#DEF'])
del data[1]
data = data.rename(columns={0:'lexeme'})
data = data[data['lexeme'] != 'lexeme']

# Label columns

cols = list(data.columns)
cols[1:7] = ['prs.1.sg', 'prs.2.sg', 'prs.3.sg', 'prs.1.pl', 'prs.2.pl', 'prs.3.pl']
cols[7:13] = ['ipfv.1.sg', 'ipfv.2.sg', 'ipfv.3.sg', 'ipfv.1.pl', 'ipfv.2.pl', 'ipfv.3.pl']
cols[13:19] = ['fut.1.sg', 'fut.2.sg', 'fut.3.sg', 'fut.1.pl', 'fut.2.pl', 'fut.3.pl']
cols[19:25] = ['cond.1.sg', 'cond.2.sg', 'cond.3.sg', 'cond.1.pl', 'cond.2.pl', 'cond.3.pl']
cols[25:31] = ['sbjv.1.sg', 'sbjv.2.sg', 'sbjv.3.sg', 'sbjv.1.pl', 'sbjv.2.pl', 'sbjv.3.pl']
cols[31:37] = ['pst.1.sg', 'pst.2.sg', 'pst.3.sg', 'pst..1.pl', 'pst.2.pl', 'pst.3.pl']
cols[37:43] = ['pst,sbjv.1.sg', 'pst,sbjv.2.sg', 'pst,sbjv.3.sg', 'pst,sbjv.1.pl', 'pst,sbjv.2.pl', 'pst,sbjv.3.pl']
cols[43:46] = ['imp.2.sg', 'imp.1.pl', 'imp.2.pl']
cols[46] = 'inf'
cols[47] = 'prs,ptcp'
cols[48] = 'pst,ptcp.m.sg'
cols[49] = 'pst,ptcp.m.pl'
cols[50] = 'pst,ptcp.f.sg'
cols[51] = 'pst,ptcp.f.pl'
data.columns = cols

# Save

data = pd.melt(data, id_vars='lexeme', var_name='features', value_name='form')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data.to_csv('french.dat.gz', compression='gzip', index=False, header=False, sep='\t', columns=['form', 'lexeme', 'features'])
