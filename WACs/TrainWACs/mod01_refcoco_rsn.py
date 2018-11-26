# coding: utf-8
'''
Train a (set of) WAC models. Modified template for a specific parameter
setting.
'''

from __future__ import division
import sys
import argparse
import ConfigParser
import codecs
import json
import os
from os.path import isfile
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.append('../../Utils')
from utils import print_timestamped_message
sys.path.append('../WAC_utils')
from wac_utils import filter_X_by_filelist, filter_refdf_by_filelist
from wac_utils import filter_relational_expr
from wac_utils import create_word2den, make_X_id_index, make_mask_matrix
from wac_utils import train_this_word

# The first features in the image feature Xs encode the region ID
ID_FEATS = 3

N_JOBS = 2  # how many threads to run in parallel during training


def main(config):
    basename = os.path.splitext(os.path.basename(__file__))[0]
    print_timestamped_message('Starting to train model %s'
                              % (basename))

    outfile_base = config.get('runtime', 'out_dir') + '/' + basename
    if isfile(outfile_base + '.npz'):
        print '%s exists. Will not overwrite. ABORTING.' % (outfile_base + '.npz')
        return

    # Model description:
    model = {
        'rcorp': 'refcoco',        # ref corpus
        'cnn': 'rsn50-flatten_1',  # CNN used for vision feats
        'rel':   'excl',           # exclude relational expressions
        'wrdl':  'min',            # wordlist: minimal n occurrences...
        'wprm':  40,               # ... 40 times
        'clsf':  'logreg-l1',      # logistic regression, l1 regularized
        'nneg':  20000,            # maximally 20k neg instances
        'nsrc':  'randmax',        # ... randomly selected
        'notes': ''
    }
    classifier = linear_model.LogisticRegression
    classf_params = {'penalty': 'l1', 'warm_start': True}

    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'

    # ========================= DATA =================================
    print_timestamped_message('loading up data.', indent=4)

    with open(preproc_path + 'refcoco_splits.json', 'r') as f:
        rc_splits = json.load(f)

    # Image features
    X = np.load(feats_path + 'mscoco_bbdf_rsn50-flatten_1.npz')['arr_0']
    X_t = filter_X_by_filelist(X, rc_splits['train'])

    # Referring expressions
    refcoco_refdf = pd.read_json(preproc_path + 'refcoco_refdf.json.gz',
                                 typ='frame', orient='split',
                                 compression='gzip')
    refdf_train = filter_refdf_by_filelist(refcoco_refdf, rc_splits['train'])

    refdf_train = filter_relational_expr(refdf_train)

    # ======================= Intermediate ==============================
    print_timestamped_message('creating intermediate data structures',
                              indent=4)
    word2den = create_word2den(refdf_train)
    X_idx = make_X_id_index(X_t)
    mask_matrix = make_mask_matrix(X_t, X_idx, word2den, word2den.keys())

    # ======================= Wordlist ==============================
    print_timestamped_message('selecting words to train models for',
                              indent=4)
    min_freq = model['wprm']
    counts = mask_matrix.sum(axis=1)
    wordlist = np.array(word2den.keys())[counts > min_freq]

    # ======================= TRAIN ==============================
    print_timestamped_message('and training the WACs!',
                              indent=4)

    wacs = Parallel(n_jobs=N_JOBS, require='sharedmem', prefer='threads')\
                   (delayed(train_this_word)(X_t, word2den, mask_matrix,
                                             model['nneg'],
                                             classifier, classf_params,
                                             this_word)
                    for this_word in wordlist[:10])
    print ''  # newline, because train_this_word prints . as progress bar

    # ======================= SAVE ==============================
    print_timestamped_message('writing to disk',
                              indent=4)
    weight_matrix = np.stack([np.append(this_wac.coef_, this_wac.intercept_)
                              for this_wac in [w[3] for w in wacs]])
    wordinfo = [e[:-1] for e in wacs]
    with open(outfile_base + '.json', 'w') as f:
        json.dump((model, wordinfo), f)
    np.savez_compressed(outfile_base + '.npz', weight_matrix)

    print_timestamped_message('DONE!')


#
#
#
# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a (set of) WAC model(s)')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file specifying data paths.
                        default: '../../Config/default.cfg' ''',
                        default='../../Config/default.cfg')
    parser.add_argument('-o', '--out_dir',
                        help='''
                        where to put the resulting files.
                        default: './ModelsOut' ''')
    args = parser.parse_args()

    config = ConfigParser.SafeConfigParser()

    try:
        with codecs.open(args.config_file, 'r', encoding='utf-8') as f:
            config.readfp(f)
    except IOError:
        print 'no config file found at %s' % (args.config_file)
        sys.exit(1)

    if args.out_dir:
        out_dir = args.out_dir
    elif config.has_option('DSGV-PATHS', 'train_out_dir'):
        out_dir = config.get('DSGV-PATHS', 'train_out_dir')
    else:
        out_dir = './ModelsOut'

    config.add_section('runtime')
    config.set('runtime', 'out_dir', out_dir)

    main(config)
