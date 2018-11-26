# coding: utf-8
'''
Some utility functions for training models. Actual training is triggered
by the scripts that define the models.
'''

from __future__ import division
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle
import sys

# The first features in the image feature Xs encode the region ID
ID_FEATS = 3

RELWORDS = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'front of',
            'right of',
            'left of',
            'ontop of',
            'next to',
            'middle of']


def filter_X_by_filelist(X, filelist):
    tmp_df = pd.DataFrame(X)
    return np.array(tmp_df[tmp_df.iloc[:, 1].isin(filelist)])


def filter_refdf_by_filelist(refdf, filelist):
    return pd.merge(refdf, pd.DataFrame(filelist, columns=['image_id']))


def is_relational(expr):
    for rel in RELWORDS:
        if rel in expr:
            return True
    return False


def filter_relational_expr(refdf):
    '''Filter out refexps with relational expressions.'''
    return refdf[~(refdf['refexp'].apply(is_relational))]


def create_word2den(refdf, refcol='refexp', regcol='region_id'):
    '''Given refdf, returns dict words to occurrences (id triples).

    Arguments:
    refdf -- the DF with the expressions

    Keyword Arguments:
    refcol    -- the column of the expressions
    region_id -- the third ID column (not always called "region_id"
    '''
    word2den = defaultdict(list)
    for _, row in refdf.iterrows():
        exprlist = row[refcol].split()
        # TODO: Could take filter function that filters out some occurences.
        #   E.g., tagger that tags whole expression & returns only the nouns.
        for word in exprlist:
            word2den[word].append((row['i_corpus'],
                                   row['image_id'],
                                   row[regcol]))
    return {k: list(set(v)) for k, v in word2den.items()}


def make_X_id_index(X, id_feats=ID_FEATS):
    '''Map ID_FEATS from matrix to index into matrix, for faster access'''
    return dict(zip([tuple(e) for e in X[:, :id_feats].astype(int).tolist()], range(len(X))))


def make_mask_matrix(X, X_idx, word2den, wordlist):
    '''Create for each word a mask vector into X, to get occurrences.

    The mask vector can be used to index X, to get all images
    that are in denotation of the word.

    Arguments:
    X      -- the feature matrix (images x features)
    X_idx  -- the output of make_X_id_index()
    word2den - dict linking words to occurrences
    wordlist - list of words for which to create masks
    '''

    mask_matrix = []
    for this_word in wordlist:
        this_word_vec = np.zeros(len(X))
        if this_word in word2den:
            this_word_vec[[X_idx[i] for i in word2den[this_word] if i in X_idx]] = 1
        mask_matrix.append(this_word_vec)
    mask_matrix = np.array(mask_matrix, dtype=bool)
    return mask_matrix


def get_X_for_word(X, word2den, mask_matrix, word, neg_max=20000):
    '''
    Get subset of X from denotation of word (or from complement, for negative examples.

    Keyword Arguments:
    - neg_max  -- if 0, no negative instances. If 'balanced', as many as positive. If positive n, capped at that number. If None, no limit.
    '''
    if word not in word2den:
        # raise ValueError("No mask available for this word! (%s)" % (word))
        print "Error!! No mask available for this word! (%s)" % (word)
        return None
    this_mask = mask_matrix[word2den.keys().index(word)]
    X_pos = X[this_mask, ID_FEATS:]
    y_pos = np.ones(len(X_pos), dtype=int)

    if neg_max == 0:
        return X_pos, y_pos

    if neg_max == 'balanced':
        neg_max = len(y_pos)

    neg_indx = np.arange(mask_matrix.shape[1])[~this_mask]
    np.random.shuffle(neg_indx)
    X_neg = X[neg_indx[:neg_max], ID_FEATS:]
    y_neg = np.zeros(len(X_neg), dtype=int)

    X_out = np.concatenate([X_pos, X_neg], axis=0)
    y_out = np.concatenate([y_pos, y_neg])
    return shuffle(X_out, y_out)


def train_this_word(X, word2den, mask_matrix, neg_max,
                    classifier, classf_params, this_word):
    X_this_w, y_this_w = get_X_for_word(X, word2den,
                                        mask_matrix, this_word,
                                        neg_max=neg_max)
    # print this_word, X_this_w.shape[0]
    print '.',
    sys.stdout.flush()
    classifier = classifier(**classf_params)
    this_wac = classifier.fit(X_this_w, y_this_w)
    return (this_word, y_this_w.sum(), len(X_this_w), this_wac)
