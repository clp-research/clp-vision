# coding: utf-8
from __future__ import division

import argparse
import re
import numpy as np
import datetime
import ConfigParser
import nltk
import codecs
import pandas as pd
import json

import sys
sys.path.append('../Utils')
from utils import icorpus_code


# ========= util functions used only here  ===========

def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()


preproc_vec = np.vectorize(preproc)

tagger = nltk.tag.perceptron.PerceptronTagger()


def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger)


def print_timestamped_message(message, indent=0):
    now = datetime.datetime.now().strftime('%Y-%m-%d @ %H:%M:%S')
    print ' ' * indent, '[ %s ] %s' % (now, message)


# ========= the actual preproc tasks, wrapped in class  ===========


class TaskFunctions(object):

    # ======= SAIAPR ========
    #
    # task-specific options:
    # --notag: don't run the tagger
    #
    def tsk_saiapr(self, config, targs):
        args, unparsed_args = targs

        print_timestamped_message('... SAIAPR', indent=4)

        refdf = pd.read_csv(config.get('SAIAPR', 'referit_path'),
                            sep='~',
                            names=['ID', 'refexp', 'regionA', 'regionB'])
        refdf['file'] = refdf['ID'].apply(lambda x:
                                          int(x.split('.')[0].split('_')[0]))
        refdf['region'] = refdf['ID'].apply(lambda x:
                                            int(x.split('.')[0].split('_')[1]))
        refdf['refexp'] = preproc_vec(refdf['refexp'])

        refdf['i_corpus'] = icorpus_code['saiapr']
        refdf['r_corpus'] = 'referit'
        refdf['image_id'] = refdf['file']
        refdf['region_id'] = refdf['region']
        refdf['rex_id'] = refdf.index.tolist()

        if '--notag' not in unparsed_args:
            refdf['tagged'] = refdf['refexp'].apply(postag)
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp', 'tagged']]
        else:
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp']]

        refdf_path = args.out_dir + '/saiapr_refdf.json.gz'
        print 'writing to disk: ', refdf_path
        refdf.to_json(refdf_path,
                      compression='gzip',
                      force_ascii=False, orient='table')

        # load and write out the splits on SAIAPR as used
        #   by Berkeley group (50/50)
        b_splits_train_p = config.get('SAIAPR', 'berkeley_splits_path') +\
            '/referit_trainval_imlist.txt'
        b_splits_test_p = config.get('SAIAPR', 'berkeley_splits_path') +\
            '/referit_test_imlist.txt'

        saiapr_train_files = np.loadtxt(b_splits_train_p, dtype=int)
        saiapr_test_files = np.loadtxt(b_splits_test_p, dtype=int)

        saiapr_berkeley_splits = {
            'test': list(saiapr_test_files),
            'train': list(saiapr_train_files)
        }

        bsplit_path = args.out_dir + '/saiapr_berkeley_10-10_splits.json'
        with open(bsplit_path, 'w') as f:
            json.dump(saiapr_berkeley_splits, f)

        # create a 90/10 split as well, to have more training data
        saiapr_train_90 = list(saiapr_train_files) +\
            list(saiapr_test_files)[:8000]
        saiapr_test_90 = list(saiapr_test_files)[8000:]
        saiapr_90_10_splits = {
            'test': saiapr_test_90,
            'train': saiapr_train_90
        }
        with open(args.out_dir + '/saiapr_90-10_splits.json', 'w') as f:
            json.dump(saiapr_90_10_splits, f)

    # == wrapper function that maps to appropriate task function ==
    def exec_task(self, task, config, targs):
        fn = getattr(self, 'tsk_' + task, None)
        if fn is None:
            print '%s is an unkown task' % task
            sys.exit(1)
        else:
            fn(config, targs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess corpora for DSG vision/language experiments')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file with data paths.
                        default: '../Config/default.cfg' ''',
                        default='../Config/default.cfg')
    parser.add_argument('-o', '--out_dir',
                        help='''
                        where to put the resulting files.,
                        default: './PreprocOut' ''',
                        default='./PreprocOut')
    parser.add_argument('task',
                        nargs='+',
                        help='''
                        task(s) to do. One or more of: 'saiapr'
                        ''')
    targs = parser.parse_known_args()
    args, _unparsed_args = targs

    tfs = TaskFunctions()

    config = ConfigParser.SafeConfigParser()

    try:
        with codecs.open(args.config_file, 'r', encoding='utf-8') as f:
            config.readfp(f)
    except IOError:
        print 'no config file found at %s' % (args.config_file)
        sys.exit(1)

    print_timestamped_message('starting to preprocess...')

    # TODO: 'all' task, runs all tsk_ functions.. look in dir(tfs)
    for task in args.task:
        tfs.exec_task(task, config, targs)

    print_timestamped_message('... done!')
