# coding: utf-8

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_dummy_column(X):
    # return np.hstack([X, np.ones((len(X), 1))])
    # add column to the last dimension
    added = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    # print('add column:', X.shape, X.shape[:-1] + (1,), added.shape)
    return added


def logreg(X, wacs):
    return sigmoid(np.dot(add_dummy_column(X), wacs))


def mlp(X, wacs):
    hid = X
    for n in range(len(wacs) - 1):
        print('- layer', n)
        print('in', hid.shape, wacs[n].shape)
        hid = np.dot(add_dummy_column(hid), wacs[n])
        print('out', hid.shape)
        hid[hid < 0] = 0  # activation function: RelU

    print('- layer', n+1)
    print('in', hid.shape, wacs[-1].shape)
    # logits = np.dot(add_dummy_column(hid), wacs[-1].T) # np.squeeze(wacs[-1], axis=-1).T)
    # let's do it by hand
    hid = add_dummy_column(hid)
    final = np.squeeze(wacs[-1])
    out = []
    for i in range(hid.shape[0]):  # region
        for j in range(hid.shape[1]):  # word
            out.append(sigmoid(np.dot(hid[i, j, :], final[j])))
    print('out', np.array(out).shape)
    return np.array(out).reshape(hid.shape[:2])
    # print('out', logits.shape)
    # return np.squeeze(sigmoid(logits))


def mlp_one_wac(X, wac):
    hid = X
    for n in range(len(wac) - 1):
        hid = np.dot(add_dummy_column(hid), wac[n])
        # print(hid.shape)
        hid[hid < 0] = 0  # activation function: RelU
    logits = np.dot(add_dummy_column(hid), wac[-1])
    return np.squeeze(sigmoid(logits))


def mlp_explicit(X, wacs):
    out = []
    for wi in range(wacs[0].shape[0]):
        out.append(mlp_one_wac(X, [wac[wi, :] for wac in wacs]))
    return np.array(out).T


def mlp_einsum(X, wacs):
    this_layer = np.einsum('ri,wio->rwo', add_dummy_column(X), wacs[0])
    this_layer[this_layer < 0] = 0  # ReLU
    # inner layers
    for n in range(1, len(wacs) - 1):
        this_layer = np.einsum('rwi,wio->rwo', add_dummy_column(this_layer), wacs[n])
        this_layer[this_layer < 0] = 0  # ReLU
    # output layer
    out = np.einsum('rwi,wi->rw', add_dummy_column(this_layer), np.squeeze(wacs[-1]))
    return sigmoid(out)


def apply_wac_set_matrix(X, wacs, net=None):
    '''Apply all wacs to whole X

    Returns matrix of responses, with rows being the instances,
    and columns the wacs (words).
    '''
    if not net:
        raise ValueError('must specify network function "net="')
    return net(X, wacs)
