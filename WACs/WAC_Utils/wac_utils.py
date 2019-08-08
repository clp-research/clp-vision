# coding: utf-8
'''
Some utility functions for training models. Actual training is triggered
by the scripts that define the models.
'''

from __future__ import division
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from itertools import permutations

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
    if type(X) == np.ndarray:
        tmp_df = pd.DataFrame(X)
        return np.array(tmp_df[tmp_df.iloc[:, 1].isin(filelist)])
    else:  # assume that X is a dask array
        image_id_list = X[:, 1].compute()
        train_mask = np.isin(image_id_list, filelist)
        return X[train_mask]


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


def make_X_id_index(X, id_feats=ID_FEATS, cast_to_int=True):
    '''Map ID_FEATS from matrix to index into matrix, for faster access'''

    # 2019-08-07: pretty inelegant, but an np.array can only have
    # one type, and for corpora where there are subregions,
    # we need to have the region id as float...
    # I don't want to test which is the case here, and I also didn't want
    # to change this completely, as this may have negative consequences
    # for old code which may rely on the keys being integers...
    if cast_to_int:
        if type(X) == np.ndarray:
            return dict(zip([tuple(e) for e in X[:, :id_feats].astype(int).tolist()], range(len(X))))
        else:  # assume that it is a dask array
            return dict(zip([tuple(e) for e in X[:, :id_feats].compute().astype(int).tolist()], range(len(X))))
    else:
        if type(X) == np.ndarray:
            return dict(zip([tuple(e) for e in X[:, :id_feats].tolist()], range(len(X))))
        else:  # assume that it is a dask array
            return dict(zip([tuple(e) for e in X[:, :id_feats].compute().tolist()], range(len(X))))


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
        this_word_vec = np.zeros(len(X), dtype=np.bool)
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

    dask_flag = 0 if type(X) == np.ndarray else 1

    if word not in word2den:
        # raise ValueError("No mask available for this word! (%s)" % (word))
        print("Error!! No mask available for this word! (%s)" % (word))
        return None
    this_mask = mask_matrix[list(word2den.keys()).index(word)]
    X_pos = X[this_mask, ID_FEATS:] if not dask_flag else X[this_mask, ID_FEATS:].compute()
    y_pos = np.ones(len(X_pos), dtype=int)

    # print('made it here!', X_pos.shape)

    if type(neg_max) is int or type(neg_max) is str:
        if neg_max == 0:
            return X_pos, y_pos

        if neg_max == 'balanced':
            neg_max = len(y_pos)

        neg_indx = np.arange(mask_matrix.shape[1])[~this_mask]
        np.random.shuffle(neg_indx)
        neg_indx = neg_indx[:neg_max]
        neg_indx = np.sort(neg_indx)   # for performance reasons, makes dask access faster
        X_neg = X[neg_indx, ID_FEATS:] if not dask_flag else X[neg_indx, ID_FEATS:].compute()
    else:
        X_neg = neg_max  # pass in X_neg from outside...
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
    print('.', end='')
    sys.stdout.flush()
    classifier = classifier(**classf_params)
    this_wac = classifier.fit(X_this_w, y_this_w)
    return (this_word, y_this_w.sum(), len(X_this_w), this_wac)


def make_X_img_index(X, image_id_idx=1):
    '''Gets position in X for each image ID / map from imageID to list of indices.

    Goes through X and adds current row to list for current image.
    '''
    X_image_index = defaultdict(list)
    for image_id, index in zip(X[:, image_id_idx], np.arange(len(X))):
        X_image_index[int(image_id)].append(index)
    return X_image_index


def make_Xneg_pairs(X_img_idx):
    '''Gets all pairs of index positions, for objects from the same image.'''
    Xneg_pairs = []
    for _image_id, object_ids in list(X_img_idx.items()):
        Xneg_pairs.extend(permutations(object_ids, 2))
    return Xneg_pairs


def get_X_for_rel(X, X_idx, Xnegpairs, r2d, rel, neg_min=20000, neg_factor=10, ffunc=lambda x: x):
    Xout = []
    not_available = 0

    for argA, argB in r2d[rel]:
        # print(argA, argB)
        # print(X[X_idx[argA], ID_FEATS:])
        try:
            Xout.append((ffunc(np.concatenate([X[X_idx[argA], ID_FEATS:], X[X_idx[argB], ID_FEATS:]]))))
        except:
            # print('pair not in index??', argA, argB)
            not_available += 1
            continue
    # print('for %d (of %d = %.2f%%) pairs, features were not available...' % (not_available,
    #                                                                       len(r2d[rel]),
    #                                                                       not_available / len(r2d[rel]) * 100))
    Ypos = np.ones(len(Xout))
    if neg_min == 0:
        return np.vstack(Xout), Ypos
    if neg_min == 'balanced':
        n_neg = len(Xout)
    else:
        n_neg = max(neg_min, min(len(Xout) * neg_factor, int(len(X) / neg_factor)))

    this_Xnegpairs = [Xnegpairs[i] for i in np.random.randint(len(Xnegpairs), size=n_neg)]
    Xneg = np.vstack([ffunc(np.concatenate([X[argA, ID_FEATS:],
                                            X[argB, ID_FEATS:]])) for argA, argB in this_Xnegpairs])

    # TODO: could hard-code potentially more informative features, like IoU, distance of centroids, angle
    #   polar coordinates...
    # TODO: could also sample (some) negative instances from pairs of other relations...
    #  more informative?

    Yneg = np.zeros(len(Xneg))

    thisX_full, thisY_full = shuffle(np.concatenate([Xout, Xneg]), np.concatenate([Ypos, Yneg]))

    return thisX_full, thisY_full


def intersectbb(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    if x1 <= x2:
        w = x1 + w1 - x2
        if x2 + w2 < x1 + w1:
            w -= x1+w1 - (x2+w2)
    else:
        w = x2 + w2 - x1
        if x2+w2 > x1+w1:
            w -= x2+w2 - (x1+w1)

    if y1 <= y2:
        h = y1 + h1 - y2
        if y2+h2 < y2+h1:
            h -= y1+h1 - (y2+h2)
    else:
        h = y2 + h2 - y1
        if y2+h2 > y1+h1:
            h -= y2+h2 - (y1+h1)
    if np.min([w, h]) < 0:
        inter = 0
    else:
        inter = w * h
    return inter


def intoveru(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    inter = intersectbb(bb1, bb2)
    union = w1 * h1 + w2 * h2 - inter
    # print bb1, bb2, inter / union
    # return inter, union, inter / union
    return inter / union


def center_point(x1, y1, x2, y2):
    return (x2-x1)/2, (y2-y1)/2


def dist_angle(pA, pB):
    (xA, yA), (xB, yB) = pA, pB
    w = xB - xA
    h = yB - yA
    dist = np.sqrt(w**2 + h**2)
    # angle = np.arcsin(w/dist)
    angle = np.arctan2(w, h)
    return dist, angle


def train_this_relation(X, X_idx, Xnegpairs, r2d,
                        neg_min, neg_factor, classifier, classf_params,
                        this_rel, ffunc=lambda x: x):
    thisX, thisY = get_X_for_rel(X, X_idx, Xnegpairs, r2d, this_rel,
                                 neg_min, neg_factor, ffunc=ffunc)
    if not len(thisX):
        print('can not train this word, no training data', this_rel)
        return (None, None, None, None)

    this_wac = classifier(**classf_params)
    this_wac.fit(thisX, thisY)

    # print(this_rel, this_wac.score(thisX, thisY))

    return (this_rel, thisY.sum(), len(thisX), this_wac)


def ffunc_ext(vector):
    x1rA, y1rA, x2rA, y2rA, areaA, ratioA, distanceA = vector[:7]
    x1rB, y1rB, x2rB, y2rB, areaB, ratioB, distanceB = vector[7:]
    # print(x1rA)
    iou = intoveru([x1rA, y1rA, x2rA-x1rA, y2rA-y1rA], [x1rB, y1rB, x2rB-x1rB, y2rB-y1rB])
    dist, angle = dist_angle(center_point(x1rA, y1rA, x2rA, y2rA), center_point(x1rB, y1rB, x2rB, y2rB))
    return np.array([x1rA, y1rA, x2rA, y2rA, x1rB, y1rB, x2rB, y2rB,
                     areaA, distanceA, areaB, distanceB, iou, dist, angle])
