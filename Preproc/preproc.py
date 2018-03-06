# coding: utf-8
'''
Preprocess the original image / text corpora so that information
is represented in a uniform format (pandas DataFrames, serialized
to json)
'''

# TODO:
# - make notag and nocompression official arguments [DONE]

from __future__ import division

import argparse
import re
import ConfigParser
import codecs
import json
import cPickle as pickle
import logging
from itertools import chain

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import nltk
import pandas as pd

from tqdm import tqdm

import sys
sys.path.append('../Utils')
from utils import icorpus_code, saiapr_image_filename, get_saiapr_bb
from utils import print_timestamped_message


# ========= util functions used only here  ===========

def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()


preproc_vec = np.vectorize(preproc)

tagger = nltk.tag.perceptron.PerceptronTagger()


def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger)


# ========= the actual preproc tasks, wrapped in class  ===========


class TaskFunctions(object):

    def __init__(self, targs, config):
        args, unparsed_args = targs
        self.args = args
        self.unparsed_args = unparsed_args
        self.config = config

    # == wrapper function that maps to appropriate task function ==
    def exec_task(self, task):
        fn = getattr(self, 'tsk_' + task, None)
        if fn is None:
            print '%s is an unkown task' % task
            sys.exit(1)
        else:
            fn()

    # static methods
    @staticmethod
    def _dumpDF(refdf, path_base, args):
        if args.nocompression:
            print 'writing to disk: ', path_base
            refdf.to_json(path_base,
                          force_ascii=False, orient='split')
        else:
            refdf_path = path_base + '.gz'
            print 'writing to disk: ', refdf_path
            refdf.to_json(refdf_path,
                          compression='gzip',
                          force_ascii=False, orient='split')

    # ======= SAIAPR ========
    #
    # task-specific options:
    #
    def tsk_saiapr(self):
        config = self.config
        args = self.args
        # unparsed_args = self.unparsed_args

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

        if args.notag:
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp']]
        else:
            refdf['tagged'] = refdf['refexp'].apply(postag)
            refdf = refdf[['i_corpus', 'image_id', 'region_id', 'r_corpus',
                           'rex_id', 'refexp', 'tagged']]

        TaskFunctions._dumpDF(refdf,
                              args.out_dir + '/saiapr_refdf.json', args)

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

    # ======= RefCoco and RefCocoPlus ========
    #
    # task-specific options:
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

        if args.notag:
            refcoco_fin = refcocodf_tmp[['i_corpus', 'image_id', 'region_id',
                                         'r_corpus', 'rex_id',
                                         'refexp']]
        else:
            refcocodf_tmp['tagged'] = refcocodf_tmp['refexp'].apply(postag)
            refcoco_fin = refcocodf_tmp[['i_corpus', 'image_id', 'region_id',
                                         'r_corpus', 'rex_id',
                                         'refexp', 'tagged']]

        TaskFunctions._dumpDF(refcoco_fin,
                              args.out_dir + '/' + outbase + '.json',
                              args)

        if outbase == 'refcoco_refdf':
            # write out the suggested splits from ReferIt team
            #  here we have more than just train and val
            refcoco_splits = {}
            for part in refcocodf['split'].value_counts().index:
                this_filelist = list(set(refcocodf[refcocodf['split'] == part]
                                         ['image_id'].tolist()))
                refcoco_splits[part] = this_filelist

            with open(args.out_dir + '/refcoco_splits.json', 'w') as f:
                json.dump(refcoco_splits, f)

    def tsk_refcoco(self):
        config = self.config
        targs = self.args, self.unparsed_args

        print_timestamped_message('... RefCoco', indent=4)

        refcoco_path = config.get('REFCOCO', 'refcoco_path')

        TaskFunctions._process_refcoco(refcoco_path,
                                       'refcoco_refdf', targs)

    def tsk_refcocoplus(self):
        config = self.config
        targs = self.args, self.unparsed_args

        print_timestamped_message('... RefCocoPlus', indent=4)

        refcocoplus_path = config.get('REFCOCO_PLUS', 'refcocoplus_path')

        TaskFunctions._process_refcoco(refcocoplus_path,
                                       'refcocoplus_refdf', targs)

    # ======= GoogleCocoRefExp ========
    #
    # task-specific options:
    #
    def tsk_grex(self):
        config = self.config
        args = self.args
        # unparsed_args = self.unparsed_args

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

        if args.notag:
            gexdf = gexdf[['i_corpus', 'image_id', 'region_id',
                           'r_corpus', 'rex_id', 'refexp']]
        else:
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

        TaskFunctions._dumpDF(gexdf, args.out_dir + '/grex_refdf.json', args)


    # ======= SAIAPR bounding boxes ========
    #
    # task-specific options:
    #
    def tsk_saiaprbb(self):
        config = self.config
        args = self.args

        print_timestamped_message('... SAIAPR Bounding Boxes', indent=4)

        featmat = scipy.io.loadmat(config.get('SAIAPR', 'saiapr_featmat'))
        X = featmat['X']

        # get all the bounding boxes for SAIAPR regions
        checked = {}
        outrows = []
        this_corpus = icorpus_code['saiapr']

        for n, row in enumerate(tqdm(X)):

            this_image_id = int(row[0])
            this_region_id = int(row[1])
            this_category = int(row[-1])
            # Skip over b/w images. Test only once for each image.
            if checked.get(this_image_id) == 'skip':
                continue
            elif checked.get(this_image_id) != 'checked':
                img = plt.imread(saiapr_image_filename(config, this_image_id))
                checked[this_image_id] = 'checked'
                if len(img.shape) != 3:
                    logging.info('skipping image %d' % (this_image_id))
                    continue
            this_bb = get_saiapr_bb(config, this_image_id, this_region_id)
            if np.min(np.array(this_bb)) < 0:
                logging.info('skipping bb for %d %d' %
                             (this_image_id, this_region_id))
                continue
            outrows.append((this_corpus, this_image_id,
                            this_region_id, this_bb, this_category))

        bbdf_saiapr = pd.DataFrame(outrows,
                                   columns=('i_corpus image_id ' +
                                            'region_id bb cat').split())

        self._dumpDF(bbdf_saiapr, args.out_dir + '/saiapr_bbdf.json', args)

    # ======= MSCOCO bounding boxes ========
    # task-specific options:
    #
    def tsk_mscocobb(self):

        config = self.config
        args = self.args

        print_timestamped_message('... MSCOCO Bounding Boxes', indent=4)

        mscoco_path = config.get('MSCOCO', 'mscoco_path')

        with open(args.out_dir + '/refcoco_splits.json', 'r') as f:
            refcoco_splits = json.load(f)

        with open(args.out_dir + '/google_refexp_rexsplits.json', 'r') as f:
            grex_splits = json.load(f)

        all_coco_files = list(set(chain(*refcoco_splits.values())).union(set(chain(*grex_splits))))

        with open(mscoco_path, 'r') as f:
            coco_in = json.load(f)

        cocoandf = pd.DataFrame(coco_in['annotations'])
        file_df = pd.DataFrame(all_coco_files, columns=['image_id'])

        cocoandf_reduced = pd.merge(cocoandf, file_df)

        bbdf_coco = cocoandf_reduced[['image_id', 'id', 'bbox', 'category_id']]
        bbdf_coco['i_corpus'] = icorpus_code['mscoco']

        bbdf_coco.columns = 'image_id region_id bb cat i_corpus'.split()
        bbdf_coco = bbdf_coco['i_corpus image_id region_id bb cat'.split()]

        self._dumpDF(bbdf_coco, args.out_dir + '/mscoco_bbdf.json', args)


    # ======= MSCOCO bounding boxes ========
    # task-specific options:
    #
    def tsk_grexbb(self):
            
        config = self.config
        args = self.args

        print_timestamped_message('... COCORex Bounding Boxes', indent=4)

        grex_path = config.get('GREX', 'grex_base') +\
                    '/google_refexp_train_201511_coco_aligned.json'
        with open(grex_path, 'r') as f:
            grex_json = json.load(f)

        gimdf = pd.DataFrame(grex_json['images']).T
        
        with open(args.out_dir + '/refcoco_splits.json', 'r') as f:
            refcoco_splits = json.load(f)

        with open(args.out_dir + '/google_refexp_rexsplits.json', 'r') as f:
            grex_splits = json.load(f)
     
        refcoco_testfiledf = pd.DataFrame(list(chain(refcoco_splits['testA'],
                                                     refcoco_splits['testB'],
                                                     refcoco_splits['val'])),
                                          columns=['image_id'])

        gimdf_reduced = pd.merge(gimdf, refcoco_testfiledf)

        rows = []
        this_i_corpus = icorpus_code['mscoco_grprops']
        for n, row in tqdm(gimdf_reduced.iterrows()):
            bbs = row['region_candidates']
            this_image_id = row['image_id']
            for k, this_bbs in enumerate(bbs):
                this_bb = this_bbs['bounding_box']
                this_cat = this_bbs['predicted_object_name']
                rows.append([this_i_corpus, this_image_id, k, this_bb, this_cat])


        bbdf_cocorprop = pd.DataFrame(rows,
                                      columns='i_corpus image_id region_id bb cat'.split())

        self._dumpDF(bbdf_cocorprop, 'PreProcOut/cocogrprops_bbdf.json', args)



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
    parser.add_argument('-n', '--nocompression',
                        help='''
                        don't use compression when writing out DFs''',
                        action='store_true')
    parser.add_argument('-t', '--notag',
                        help='''
                        don't POS-tag the expressions''',
                        action='store_true')
    parser.add_argument('task',
                        nargs='+',
                        choices=['saiapr', 'refcoco', 'refcocoplus',
                                 'grex', 'saiaprbb', 'mscocobb', 'grexbb', 'all'],
                        help='''
                        task(s) to do. Choose one or more.
                        'all' runs all tasks.''')

    targs = parser.parse_known_args()
    args, unparsed_args = targs

    print "treated as task-specific parameters: ", unparsed_args

    config = ConfigParser.SafeConfigParser()

    try:
        with codecs.open(args.config_file, 'r', encoding='utf-8') as f:
            config.readfp(f)
    except IOError:
        print 'no config file found at %s' % (args.config_file)
        sys.exit(1)

    tfs = TaskFunctions(targs, config)

    if 'all' in args.task:
        available_tasks = [this_method.replace('tsk_', '')
                           for this_method in dir(tfs)
                           if this_method.startswith('tsk_')]
        print 'I will run all of:', available_tasks
        args.task = available_tasks

    print_timestamped_message('starting to preprocess...')

    # TODO: 'all' task, runs all tsk_ functions.. look in dir(tfs)
    for task in args.task:
        tfs.exec_task(task)

    print_timestamped_message('... done!')
