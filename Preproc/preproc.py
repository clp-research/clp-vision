# coding: utf-8
'''
Preprocess the original image / text corpora so that information
is represented in a uniform format (pandas DataFrames, serialized
to json)
'''

# TODO:
# - make notag and nocompression official arguments

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
import cPickle as pickle

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
    # --nocompression: don't compress the output json
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

        if '--notag' in unparsed_args:
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp']]
        else:
            refdf['tagged'] = refdf['refexp'].apply(postag)
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp', 'tagged']]

        if '--nocompression' in unparsed_args:
            refdf_path = args.out_dir + '/saiapr_refdf.json'
            print 'writing to disk: ', refdf_path
            refdf.to_json(refdf_path,
                          force_ascii=False, orient='table')
        else:
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

    # ======= RefCoco and RefCocoPlus ========
    #
    # task-specific options:
    # --notag: don't run the tagger
    # --nocompression: don't compress the output json
    #
    @staticmethod
    def _process_refcoco(inpath, outbase, targs):
        args, unparsed_args = targs

        with open(inpath, 'r') as f:
            refcoco = pickle.load(f)

        refcocodf = pd.DataFrame(refcoco)

        refdf_list = []
        for n, this_row in refcocodf.iterrows():
            this_file = this_row['image_id']
            this_region = this_row['ann_id']
            for this_sentence, this_rexid in zip(this_row['sentences'],
                                                 this_row['sent_ids']):
                this_sentence_sent = this_sentence['sent']
                refdf_list.append((this_file, this_region,
                                   this_sentence_sent, this_rexid))

        refcocodf_tmp = pd.DataFrame(refdf_list,
                                     columns=('image_id region_id ' +
                                              'refexp rex_id').split())

        refcocodf_tmp['i_corpus'] = icorpus_code['mscoco']
        refcocodf_tmp['r_corpus'] = 'refcoco'

        if '--notag' in unparsed_args:
            refcoco_fin = refcocodf_tmp[['i_corpus', 'image_id', 'region_id',
                                         'r_corpus', 'rex_id',
                                         'refexp']]
        else:
            refcocodf_tmp['tagged'] = refcocodf_tmp['refexp'].apply(postag)
            refcoco_fin = refcocodf_tmp[['i_corpus', 'image_id', 'region_id',
                                         'r_corpus', 'rex_id',
                                         'refexp', 'tagged']]

        if '--nocompression' in unparsed_args:
            refdf_path = args.out_dir + '/' + outbase + '.json'
            print 'writing to disk: ', refdf_path
            refcoco_fin.to_json(refdf_path,
                                force_ascii=False, orient='table')
        else:
            refdf_path = args.out_dir + '/' + outbase + '.json.gz'
            print 'writing to disk: ', refdf_path
            refcoco_fin.to_json(refdf_path,
                                compression='gzip',
                                force_ascii=False, orient='table')

        if outbase == 'refcoco':
            # write out the suggested splits from ReferIt team
            #  here we have more than just train and val
            refcoco_splits = {}
            for part in refcocodf['split'].value_counts().index:
                this_filelist = list(set(refcocodf[refcocodf['split'] == part]
                                         ['image_id'].tolist()))
                refcoco_splits[part] = this_filelist

            with open(args.out_dir + '/refcoco_splits.json', 'w') as f:
                json.dump(refcoco_splits, f)

    def tsk_refcoco(self, config, targs):
        args, unparsed_args = targs

        print_timestamped_message('... RefCoco', indent=4)

        refcoco_path = config.get('REFCOCO', 'refcoco_path')

        TaskFunctions._process_refcoco(refcoco_path,
                                       'refcoco_refdf', targs)

    def tsk_refcocoplus(self, config, targs):
        args, unparsed_args = targs

        print_timestamped_message('... RefCocoPlus', indent=4)

        refcocoplus_path = config.get('REFCOCO_PLUS', 'refcocoplus_path')

        TaskFunctions._process_refcoco(refcocoplus_path,
                                       'refcocoplus_refdf', targs)

    # ======= GoogleCocoRefExp ========
    #
    # task-specific options:
    # --notag: don't run the tagger
    # --nocompression: don't compress the output json
    #
    def tsk_grex(self, config, targs):
        args, unparsed_args = targs

        print_timestamped_message('... GoogleCOCOrex', indent=4)

        gjson_p = config.get('GREX', 'grex_base') +\
            '/google_refexp_train_201511_coco_aligned.json'
        with open(gjson_p, 'r') as f:
            gexp = json.load(f)
        gexan = pd.DataFrame(gexp['annotations']).T
        gexrex = pd.DataFrame(gexp['refexps']).T

        gjson_p = config.get('GREX', 'grex_base') +\
            '/google_refexp_val_201511_coco_aligned.json'
        with open(gjson_p, 'r') as f:
            gexpv = json.load(f)
        gexanv = pd.DataFrame(gexpv['annotations']).T
        gexrexv = pd.DataFrame(gexpv['refexps']).T

        gexanfull = pd.concat([gexan, gexanv])
        gexrexfull = pd.concat([gexrex, gexrexv])

        outrows = []
        for n, row in gexanfull.iterrows():
            this_image_id = row['image_id']
            this_anno_id = row['annotation_id']
            this_refexp_ids = row['refexp_ids']
            for this_refexp_id in this_refexp_ids:
                this_refexp = gexrexfull[
                    gexrexfull['refexp_id'] == this_refexp_id]['raw'][0]
                this_refexp = re.sub('[\.\,\?;]+', '', this_refexp).lower()
                this_refexp = this_refexp.encode('UTF-8')
                outrows.append((this_image_id, this_anno_id,
                                this_refexp, this_refexp_id))

        gexdf = pd.DataFrame(outrows,
                             columns=('image_id region_id ' +
                                      'refexp rex_id').split())
        gexdf['i_corpus'] = icorpus_code['mscoco']
        gexdf['r_corpus'] = 'grex'

        gexdf['tagged'] = gexdf['refexp'].apply(postag)

        gexdf = gexdf[['i_corpus', 'image_id', 'region_id',
                       'r_corpus', 'rex_id', 'refexp', 'tagged']]

        # write out the splits as suggested by Google team
        #   NB: The splits here contain *refexp_ids*, not image_ids!
        gexsplits = {
            'train': gexrex['refexp_id'].tolist(),
            'val': gexrexv['refexp_id'].tolist()
            }

        with open(args.out_dir + '/google_refexp_rexsplits.json', 'w') as f:
            json.dump(gexsplits, f)

        if '--nocompression' in unparsed_args:
            refdf_path = args.out_dir + '/saiapr_refdf.json'
            print 'writing to disk: ', refdf_path
            refdf.to_json(refdf_path,
                          force_ascii=False, orient='table')
        else:
            refdf_path = args.out_dir + '/saiapr_refdf.json.gz'
            print 'writing to disk: ', refdf_path
            refdf.to_json(refdf_path,
                          compression='gzip',
                          force_ascii=False, orient='table')


# ======== MAIN =========
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
