#!/usr/bin/env python
"""
Baseline system for the CoNLL-SIGMORPHON 2017 Shared Task.

Based on Mans Hulden's implementation at:

https://github.com/sigmorphon/conll2017

"""

from collections import defaultdict
from functools import wraps
from multiprocessing import Pool
import time

import pandas as pd
import numpy as np
import regex as re

def hamming(s, t):
    return sum([1 for x, y in zip(s, t) if x != y])

def halign(s, t):
    """Align two lists by Hamming distance."""
    slen = len(s)
    tlen = len(t)
    minscore = slen + tlen + 1
    for upad in range(0, tlen + 1):
        upper = ['_'] * upad + s + (tlen - upad) * ['_']
        lower = slen * ['_'] + t
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    for lpad in range(0, slen + 1):
        upper = tlen * ['_'] + s
        lower = (slen - lpad) * ['_'] + t + ['_'] * lpad
        score = hamming(upper, lower)
        if score < minscore:
            bu = upper
            bl = lower
            minscore = score

    zipped = list(zip(bu, bl))
    newin = [i for i, o in zipped if i != '_' or o != '_']
    newout = [o for i, o in zipped if i != '_' or o != '_']
    return newin, newout


def levenshtein(s, t, inscost=1.0, delcost=1.0, substcost=1.0):
    """Recursive implementation of Levenshtein, with alignments returned."""

    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * ['_'], tpast + trem, [], [], cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * ['_'], [], [], cost + len(srem)

        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost

        return min((lrec(spast + [srem[0]], tpast + [trem[0]], srem[1:], trem[1:], cost + addcost),
                    lrec(spast + ['_'], tpast + [trem[0]], srem, trem[1:], cost + inscost),
                    lrec(spast + [srem[0]], tpast + ['_'], srem[1:], trem, cost + delcost)),
                   key=lambda x: x[4])

    answer = lrec([], [], s, t, 0)
    return answer[0], answer[1], answer[4]


def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}

    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        ksr = tuple(sr)
        ktr = tuple(tr)
        if (ksr, ktr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(ksr, ktr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        hit = cache[(ksr, ktr)]
        return sp + hit[0], tp + hit[1], [], [], cost + hit[2]

    return wrap


def alignprs(lemma, form):
    """Break lemma/form into three parts:
    IN:  1 | 2 | 3
    OUT: 4 | 5 | 6
    1/4 are assumed to be prefixes, 2/5 the stem, and 3/6 a suffix.
    1/4 and 3/6 may be empty.
    """

    al = levenshtein(lemma, form, substcost=1.1)  # Force preference of 0:x or x:0 by 1.1 cost
    alemma, aform = al[0], al[1]
    # leading spaces
    lspace = max(numleadingsyms(alemma, '_'),  numleadingsyms(aform, '_'))
    # trailing spaces
    tspace = max(numtrailingsyms(alemma, '_'), numtrailingsyms(aform, '_'))
    return alemma[0:lspace], \
           alemma[lspace:len(alemma) - tspace], alemma[len(alemma) - tspace:], \
           aform[0:lspace], aform[lspace:len(alemma) - tspace], \
           aform[len(alemma) - tspace:]


def remove(s, symbol):
    return [c for c in s if c != symbol]

def get_rules(row):
    """Extract a number of suffix-change and prefix-change rules
    based on a given example lemma+inflected form."""

    #global prefbias, suffbias, allprules, allsrules

    if prefbias > suffbias:
        lemma = list(reversed(row[2]))
        form = list(reversed(row[1]))
    else:
        lemma = row[2]
        form = row[1]

    lp, lr, ls, fp, fr, fs = alignprs(lemma, form)  # Get six parts, three for in three for out

    # Suffix rules
    ins = lr + ls + [">"]
    outs = fr + fs + [">"]
    srules = set()
    for i in range(min(len(ins), len(outs))):
        srules.add((tuple(ins[i:]), tuple(outs[i:])))
    srules = {(tuple(remove(x[0], '_')), tuple(remove(x[1], '_'))) for x in srules}

    # Prefix rules
    prules = set()
    if len(lp) >= 0 or len(fp) >= 0:
        inp = ["<"] + lp
        outp = ["<"] + fp
        for i in range(0, len(fr)):
            prules.add((tuple(inp + fr[:i]), tuple(outp + fr[:i])))
            prules = {(tuple(remove(x[0], '_')), tuple(remove(x[1], '_'))) for x in prules}

    return row[3], prules, srules

def apply_rules(row):
    """Applies the longest-matching suffix-changing rule given an input
    form and the MSD. Length ties in suffix rules are broken by frequency.
    For prefix-changing rules, only the most frequent rule is chosen."""

    if prefbias > suffbias:
        lemma = list(reversed(row[2]))
        form = list(reversed(row[1]))
    else:
        lemma = row[2]
        form = row[1]
    msd = row[3]
    if msd not in allprules and msd not in allsrules:
        return lemma == form  # Haven't seen this inflection, so bail out

    base = ["<"] + lemma + [">"]

    if msd in allsrules:
        applicablerules = [(x[0], x[1], y) for x, y in allsrules[msd].items() if base[-len(x[0]):] == list(x[0])]
        if applicablerules:
            bestrule = max(applicablerules, key=lambda x: (len(x[0]), x[2], len(x[1])))
            base[-len(bestrule[0]):] = list(bestrule[1])

    if msd in allprules:
        #applicablerules = [(x[0], x[1], y) for x, y in allprules[msd].items() if contains(base, x[0])]
        applicablerules = [(x[0], x[1], y) for x, y in allprules[msd].items() if base[:len(x[0])] == list(x[0])]
        if applicablerules:
            bestrule = max(applicablerules, key=lambda x: (x[2]))
            base[:len(bestrule[0])] = list(bestrule[1])

    del base[0]
    del base[-1]
    return (base == form)

def numleadingsyms(s, symbol):
    for i,c in enumerate(s):
        if c != symbol:
            return i
    return len(s)

def numtrailingsyms(s, symbol):
    for i,c in enumerate(reversed(s)):
        if c != symbol:
            return i
    return len(s)

def check_bias(row):
    aligned = halign(row[2], row[1])
    return (numleadingsyms(aligned[0], '_') + numleadingsyms(aligned[1], '_'),
            numtrailingsyms(aligned[0], '_') + numtrailingsyms(aligned[1], '_'))

###############################################################################

prefbias, suffbias = 0, 0
allprules, allsrules = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))

def baseline(data, index, **kwargs):

    #global prefbias, suffbias

    train = data[~index]
    test = data[index]

    with Pool() as p:

        for pre, suf in p.imap_unordered(check_bias, train.itertuples(name=None)):
            prefbias = prefbias + pre
            suffbias = suffbias + suf

    with Pool() as p:
        for msd, prules, srules in p.imap_unordered(get_rules, train.itertuples(name=None)):
            for r in prules:
                allprules[msd][(r[0], r[1])] = allprules[msd][(r[0], r[1])] + 1
            for r in srules:
                allsrules[msd][(r[0], r[1])] = allsrules[msd][(r[0], r[1])] + 1

    with Pool() as p:
        results = p.map(apply_rules, test.itertuples(name=None))

    return sum(results)/len(results)*100
