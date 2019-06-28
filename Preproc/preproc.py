# coding: utf-8
'''
Preprocess the original image / text corpora so that information
is represented in a uniform format (pandas DataFrames, serialized
to json)
'''

# TODO:
# - make notag and nocompression official arguments [DONE]

from __future__ import division

from operator import itemgetter
import argparse
import re
import configparser
import xml.etree.ElementTree as ET
import json
from ijson import items
import pickle
import logging
from itertools import chain
import glob
import gzip

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import os

from tqdm import tqdm

import sys
sys.path.append('../Utils')
from utils import icorpus_code, saiapr_image_filename, get_saiapr_bb
from utils import print_timestamped_message
sys.path.append('Helpers')
from visgen_helpers import serialise_region_descr, empty_to_none
from ade_helpers import id_mask, ade_path_data, ade_annotation, get_ade_bb
#from Preproc.Helpers.cocoent_helpers import serialise_cococap

N_VISGEN_IMG = 108077
#  The number of images in the visgen set, for the progress bar


# ========= util functions used only here  ===========

def preproc(utterance):
    utterance = re.sub(r'[\.\,\?;]+', '', utterance)
    return utterance.lower()


preproc_vec = np.vectorize(preproc)

tagger = nltk.tag.perceptron.PerceptronTagger()


def postag(refexp):
    return nltk.tag._pos_tag(nltk.word_tokenize(refexp), None, tagger, lang='eng')


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
            print('%s is an unkown task' % task)
            sys.exit(1)
        else:
            fn()

    # static methods
    @staticmethod
    def _dumpDF(refdf, path_base, args):
        if args.nocompression:
            print('writing to disk: ', path_base)
            refdf.to_json(path_base,
                          force_ascii=False, orient='split')
        else:
            refdf_path = path_base + '.gz'
            print('writing to disk: ', refdf_path)
            refdf.to_json(refdf_path,
                          compression='gzip',
                          force_ascii=False, orient='split')

    # ======= SAIAPR ========
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
    def tsk_saiaprbb(self):
        config = self.config
        args = self.args

        print_timestamped_message('... SAIAPR Bounding Boxes', indent=4)

        featmat = spio.loadmat(config.get('SAIAPR', 'saiapr_featmat'))
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
    #
    def tsk_mscocobb(self):
        # 2019-03-13: I have removed the restriction to extracting
        #  only the files that are in refcoco and grex, to get
        #  all objects. I'm leaving the code commented in here for now,
        #  as I don't know right now whether this might change anything
        #  for old downstream functions.
        #
        # 2019-03-13: TODO. This should actually run over validation
        #   as well. See guesswhat.

        config = self.config
        args = self.args

        print_timestamped_message('... MSCOCO Bounding Boxes', indent=4)

        mscoco_path = config.get('MSCOCO', 'mscoco_path')

        # with open(args.out_dir + '/refcoco_splits.json', 'r') as f:
        #     refcoco_splits = json.load(f)

        # with open(args.out_dir + '/google_refexp_rexsplits.json', 'r') as f:
        #     grex_splits = json.load(f)

        # all_coco_files = list(set(chain(*refcoco_splits.values())).union(set(chain(*grex_splits))))

        # all_coco_files = [e for e in all_coco_files if type(e) is int]

        with open(mscoco_path, 'r') as f:
            coco_in = json.load(f)

        cocoandf = pd.DataFrame(coco_in['annotations'])
        # file_df = pd.DataFrame(all_coco_files, columns=['image_id'])

        # cocoandf_reduced = pd.merge(cocoandf, file_df)
        cocoandf_reduced = cocoandf

        bbdf_coco = cocoandf_reduced[['image_id', 'id', 'bbox', 'category_id']]
        bbdf_coco['i_corpus'] = icorpus_code['mscoco']

        bbdf_coco.columns = 'image_id region_id bb cat i_corpus'.split()
        bbdf_coco = bbdf_coco['i_corpus image_id region_id bb cat'.split()]

        self._dumpDF(bbdf_coco, args.out_dir + '/mscoco_bbdf.json', args)

    # ======= MSCOCO category list ========
    #
    def tsk_mscococats(self):
        config = self.config
        args = self.args

        print_timestamped_message('... MSCOCO Cats', indent=4)

        mscoco_path = config.get('MSCOCO', 'mscoco_path')

        with open(mscoco_path, 'r') as f:
            coco_in = json.load(f)

        cococatsdf = pd.DataFrame(coco_in['categories'])

        cococatsdf.columns = 'cat_id cat supercat'.split()
        cococatsdf.set_index('cat_id', drop=True, inplace=True)

        self._dumpDF(cococatsdf, args.out_dir + '/mscoco_catsdf.json', args)

    # ======= MSCOCO region proposal bounding boxes ========
    #
    def tsk_grexbb(self):

        config = self.config
        args = self.args

        print_timestamped_message('... COCORex Bounding Boxes', indent=4)

        grex_path = config.get('GREX', 'grex_base') + '/google_refexp_train_201511_coco_aligned.json'
        with open(grex_path, 'r') as f:
            grex_json = json.load(f)

        gimdf = pd.DataFrame(grex_json['images']).T
        gimdf['image_id'] = gimdf['image_id'].astype(int)

        with open(args.out_dir + '/refcoco_splits.json', 'r') as f:
            refcoco_splits = json.load(f)

        all_files = list(chain(refcoco_splits['testA'],
                               refcoco_splits['testB'],
                               refcoco_splits['val']))
        all_files = [e for e in all_files if type(e) is int]
        refcoco_testfiledf = pd.DataFrame(all_files,
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
                rows.append([this_i_corpus, this_image_id, k,
                             this_bb, this_cat])

        bbdf_cocorprop = pd.DataFrame(rows,
                                      columns='i_corpus image_id region_id bb cat'.split())

        self._dumpDF(bbdf_cocorprop, args.out_dir + '/cocogrprops_bbdf.json', args)

    # ======= MSCOCO captions ========
    #
    def tsk_mscococap(self):

        config = self.config
        args = self.args

        print_timestamped_message('... MSCOCO Captions', indent=4)

        coco_captions_path = config.get('MSCOCO', 'mscoco_base') + '/annotations/captions_train2014.json'
        with open(coco_captions_path, 'r') as f:
            coca_json = json.load(f)
        cococap_df = pd.DataFrame(coca_json['annotations'])
        cococap_df['i_corpus'] = icorpus_code['mscoco']

        self._dumpDF(cococap_df, args.out_dir + '/cococapdf.json', args)

    # ======= Visual Genome Region Descriptions ========
    #
    def tsk_visgenreg(self):
        config = self.config
        args = self.args

        print_timestamped_message('... VisualGenome Regions', indent=4)

        vgimd_path = config.get('VISGEN', 'visgen_12') + '/jsons/image_data.json'
        vgreg_path = config.get('VISGEN', 'visgen_12') + '/jsons/region_graphs.json'
        this_corpus = icorpus_code['visual_genome']

        with open(vgimd_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(iterator):
                out.append((entry['image_id'], entry['coco_id'], entry['flickr_id']))
        vg_im_df = pd.DataFrame(out,
                                columns='image_id coco_id flickr_id'.split())

        with open(vgreg_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                image_id = entry['image_id']
                image_id_lookup = vg_im_df[vg_im_df['image_id'] == image_id]
                coco_id = image_id_lookup['coco_id'].values[0]
                flickr_id = image_id_lookup['flickr_id'].values[0]
                for this_region in entry['regions']:
                    region_id = this_region['region_id']
                    phrase = this_region['phrase']
                    x, y, w, h = (this_region['x'], this_region['y'],
                                  this_region['width'], this_region['height'])

                    sreg = serialise_region_descr(this_region)
                    out.append((this_corpus, image_id, coco_id, flickr_id,
                                region_id, phrase, [x, y, w, h]) + sreg)

        vgreg_df = pd.DataFrame(out,
                                columns='i_corpus image_id coco_id flickr_id region_id phrase bb rel_ids rels pphrase'.split())

        self._dumpDF(vgreg_df, args.out_dir + '/vgregdf.json', args)

    # ======= Visual Genome Image Data ========
    #
    def tsk_visgenimg(self):
        config = self.config
        args = self.args

        print_timestamped_message('... VisualGenome Image Data', indent=4)

        vgimd_path = config.get('VISGEN', 'visgen_12') + '/jsons/image_data.json'
        this_corpus = icorpus_code['visual_genome']

        with open(vgimd_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(iterator):
                out.append((this_corpus,
                            entry['image_id'],
                            entry['coco_id'],
                            entry['flickr_id'],
                            entry['width'],
                            entry['height']))
        colnames = 'i_corpus image_id coco_id flickr_id width height'.split()
        vg_im_df = pd.DataFrame(out, columns=colnames)

        self._dumpDF(vg_im_df, args.out_dir + '/vgimgdf.json', args)

    # ======= Visual Genome Relationships ========
    #
    def tsk_visgenrel(self):
        config = self.config
        args = self.args

        corpus_id = icorpus_code['visual_genome']

        print_timestamped_message('... VisualGenome Relationships', indent=4)

        vgrel_path = config.get('VISGEN', 'visgen_12') + '/jsons/relationships.json'
        with open(vgrel_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                image_id = entry['image_id']
                for rel in entry['relationships']:
                    rel_syn = empty_to_none(rel['synsets'])
                    sub_syn = empty_to_none(rel['subject']['synsets'])
                    obj_syn = empty_to_none(rel['object']['synsets'])
                    for this_rel in rel_syn:
                        for this_sub in sub_syn:
                            for this_obj in obj_syn:
                                out.append((corpus_id,
                                            image_id,
                                            this_rel,
                                            rel['predicate'],
                                            rel['relationship_id'],
                                            this_sub,
                                            rel['subject']['object_id'],
                                            this_obj,
                                            rel['object']['object_id']))
        vgrel_df = pd.DataFrame(out,
                                columns='i_corpus image_id rel_syn predicate rel_id sub_syn sub_id obj_syn obj_id'.split())
        self._dumpDF(vgrel_df, args.out_dir + '/vgreldf.json', args)

    # ======= Visual Genome Objects ========
    #
    def tsk_visgenobj(self):
        config = self.config
        args = self.args

        corpus_id = icorpus_code['visual_genome']

        print_timestamped_message('... VisualGenome Objects', indent=4)

        vgobj_path = config.get('VISGEN', 'visgen_12') + '/jsons/objects.json'
        with open(vgobj_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                image_id = entry['image_id']
                for obj in entry['objects']:
                    syn = empty_to_none(obj['synsets'])
                    names = empty_to_none(obj['names'])
                    for this_syn in syn:
                        for this_name in names:
                            out.append((corpus_id,
                                        image_id,
                                        obj['object_id'],
                                        this_syn,
                                        this_name,
                                        (obj['x'], obj['y'], obj['w'], obj['h'])))
        vgobj_df = pd.DataFrame(out,
                                columns='i_corpus image_id obj_id syn name bb'.split())
        self._dumpDF(vgobj_df, args.out_dir + '/vgobjdf.json', args)

    # ======= Visual Genome Attributes ========
    #
    def tsk_visgenatt(self):
        config = self.config
        args = self.args

        corpus_id = icorpus_code['visual_genome']

        print_timestamped_message('... VisualGenome Attributes', indent=4)

        vgatt_path = config.get('VISGEN', 'visgen_12') + '/jsons/attributes.json'
        with open(vgatt_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                image_id = entry['image_id']
                for obj in entry['attributes']:
                    if 'attributes' not in obj:
                        continue
                    atts = obj['attributes']
                    out.append((corpus_id,
                                image_id,
                                obj['object_id'],
                                atts))
        vgatt_df = pd.DataFrame(out,
                                columns='i_corpus image_id obj_id attributes'.split())
        self._dumpDF(vgatt_df, args.out_dir + '/vgattdf.json', args)

    # ======= Visual Genome VQAs ========
    #
    def tsk_visgenvqa(self):
        config = self.config
        args = self.args

        corpus_id = icorpus_code['visual_genome']

        print_timestamped_message('... VisualGenome VQAs', indent=4)

        vgvqa_path = config.get('VISGEN', 'visgen_12') + '/jsons/question_answers.json'
        with open(vgvqa_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                for this_qa in entry['qas']:
                    out.append((corpus_id,
                                this_qa['image_id'],
                                this_qa['qa_id'],
                                this_qa['question'],
                                this_qa['answer'],
                                this_qa['q_objects'],
                                this_qa['a_objects']))

        vgvqa_df = pd.DataFrame(out,
                                columns='i_corpus image_id qa_id q a q_objs a_objs'.split())
        visgenqamap_p = config.get('VISGEN', 'visgen_12') + '/jsons/qa_to_region_mapping.json'

        with open(visgenqamap_p, 'r') as f:
            vgqamap = json.load(f)

        vgqamap = dict([(int(e[0]), e[1]) for e in vgqamap.items()])

        vgqamap_df = pd.DataFrame(vgqamap.items(), columns='qa_id region_id'.split())

        vgvqa_df = vgvqa_df.merge(vgqamap_df, how='left')

        self._dumpDF(vgvqa_df, args.out_dir + '/vgvqadf.json', args)

    # ======= Visual Genome Paragraphs ========
    #
    def tsk_visgenpar(self):
        config = self.config
        args = self.args

        corpus_id = icorpus_code['visual_genome']

        print_timestamped_message('... VisualGenome Paragraphs', indent=4)

        vgpar_path = config.get('DEFAULT', 'corpora_base') + '/visualgenome_paragraphs/paragraphs_v1.json'

        with open(vgpar_path, 'r') as f:
            par_json = json.load(f)

        par_df = pd.DataFrame(par_json)

        par_df['i_corpus'] = corpus_id
        par_df.drop('url', axis=1, inplace=True)

        self._dumpDF(par_df, args.out_dir + '/vgpardf.json', args)

    # ======= Flickr 30k Entities CapDf ========
    #
    def tsk_flickrcap(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Flickr 30k Entities Captions', indent=4)

        flckrsent_path = config.get('FLICKR', 'flickr_sentences')
        out = []
        corpus_id = icorpus_code['flickr_30k']

        for filename in os.listdir(flckrsent_path):
            sents = []
            with open(flckrsent_path+'/'+filename, 'r') as f:
                for line in f:
                    sents.append(line)

            this_id = filename.split('.')[0]

            for sentence in sents:
                row = {}
                row['i_corpus'] = corpus_id
                row['image_id'] = this_id
                row['caption_annotated'] = sentence

                temp = re.sub(r'[\]\[\n]', '', sentence)

                row['caption_raw'] = ' '.join([word.lower() for word in temp.split() if not word.startswith('/')])

                these_entities = []
                for entity in re.findall(r'\[.*?\]', sentence):
                    these_entities.append(re.search(r'#([0-9]+)/', entity).group(1))
                row['entities'] = [int(e) for e in these_entities]

                out.append(row)
        flickr_capdf = pd.DataFrame(out)

        column_order = 'i_corpus image_id caption_annotated caption_raw entities'.split()
        self._dumpDF(flickr_capdf[column_order],
                     args.out_dir + '/flickr_capdf.json', args)

    # ======= Flickr 30k Entities BBDf ========
    #
    def tsk_flickrbb(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Flickr 30k Entities Bounding Boxes', indent=4)

        flckrbb_path = config.get('FLICKR', 'flickr_annotations')

        out = []
        corpus_id = icorpus_code['flickr_30k']

        for filename in os.listdir(flckrbb_path):
            tree = ET.parse(flckrbb_path+'/'+filename)
            root = tree.getroot()

            this_id = filename.split('.')[0]

            for obj in root.findall('object'):
                if obj.find('bndbox') is not None:
                    row = {}
                    row['i_corpus'] = corpus_id
                    row['image_id'] = this_id
                    row['region_id'] = int(obj.find('name').text)

                    # need to go from top-right coordinates to width, height
                    coords = [c for c in obj.find('bndbox')]
                    x = int(coords[0].text)
                    y = int(coords[1].text)
                    w = int(coords[2].text) - x
                    h = int(coords[3].text) - y
                    row['bb'] = [x, y, w, h]

                    out.append(row)
        flickr_bbdf = pd.DataFrame(out)

        flickr_bbdf['subregion_id'] = 1
        counts = flickr_bbdf['region_id'].value_counts() > 1
        multi_objs = counts[counts == True]

        for i in multi_objs.index:
            ix = 1
            for n, row in flickr_bbdf[flickr_bbdf.region_id == i].iterrows():
                flickr_bbdf.at[n, 'subregion_id'] = ix
                ix += 1
        flickr_bbdf.subregion_id = flickr_bbdf.subregion_id.astype('float')

        column_order = 'i_corpus image_id region_id subregion_id bb'.split()
        self._dumpDF(flickr_bbdf[column_order],
                     args.out_dir + '/flickr_bbdf.json', args)

    # ======= Flickr 30k Objects DF ========
    #
    def tsk_flickrobj(self):
        args = self.args

        print_timestamped_message('... Flickr 30k Entities Objects', indent=4)

        # this requires flickr_capdf to be present in the default out dir
        flickr_capdf = pd.read_json(args.out_dir + '/flickr_capdf.json.gz',
                                    typ='frame', orient='split',
                                    compression='gzip')
        out = []
        for _, row in flickr_capdf.iterrows():
            ic, ii = row['i_corpus image_id'.split()]
            for phrase in re.findall(r'\[.*?\]', row['caption_annotated']):
                entity_markup, phrase_text = phrase.split(' ', 1)
                phrase_text = phrase_text[:-1]
                _, entity_id, cat = entity_markup.split('/', 2)
                _, entity_id = entity_id.split('#')
                out.append((ic, ii, entity_id, phrase_text, cat))
        columns = 'i_corpus image_id region_id phrase cat'.split()
        flickr_obdf = pd.DataFrame(out, columns=columns)
        self._dumpDF(flickr_obdf, args.out_dir + '/flickr_objdf.json', args)

    # ======= CUB Birds 2011 Bounding Boxes ========
    #
    def tsk_birdbb(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Caltech-UCSD Birds-200-2011 Bounding Boxes', indent=4)

        bird_basepath = config.get('CUB_BIRDS', 'birds_base')

        with open(bird_basepath+'/images.txt', 'r') as f:
            img_paths = [line.split() for line in f.readlines()]
        with open(bird_basepath+'/bounding_boxes.txt', 'r') as f:
            img_bbs = [line.split() for line in f.readlines()]
        with open(bird_basepath+'/train_test_split.txt', 'r') as f:
            splits = [line.split() for line in f.readlines()]

        pathdf = pd.DataFrame(img_paths, columns='image_id image_path'.split())
        boxdf = pd.DataFrame(img_bbs, columns='image_id x y w h'.split())
        splitdf = pd.DataFrame(splits, columns='image_id is_train'.split())

        bird_df = reduce(lambda x, y: pd.merge(x, y, on='image_id'),
                         [pathdf, boxdf, splitdf])
        bird_df['bb'] = bird_df.apply(lambda p: [int(float(num)) for num
                                                 in [p.x, p.y, p.w, p.h]], axis=1)
        bird_df['i_corpus'] = icorpus_code['cub_birds']
        bird_df['image_id'] = pd.to_numeric(bird_df['image_id'])
        bird_df['category'] = bird_df['image_path'].apply(
            lambda x: x.split('/')[0].split('.')[1])

        column_order = 'i_corpus image_id image_path category bb is_train'.split()
        cub_bbdf = bird_df[column_order]

        self._dumpDF(cub_bbdf, args.out_dir + '/cub_bbdf.json', args)

    # ======= CUB Birds 2011 Attributes ========
    #
    def tsk_birdattr(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Caltech-UCSD Birds-200-2011 Attributes', indent=4)

        bird_attrpath = config.get('CUB_BIRDS', 'birds_base') + '/attributes'

        # this requires cub_bbdf to be present in the default out dir
        cub_bbdf = pd.read_json(args.out_dir + '/cub_bbdf.json.gz',
                                typ='frame', orient='split',
                                compression='gzip')
        attr_dict = {}
        with open(bird_attrpath+'/attributes.txt', 'r') as f:
            for line in f.readlines():
                line_info = re.search(r'(\d+) (.+)::(.+)', line)
                attr_dict[line_info.group(1)] = line_info.group(2, 3)

        with open(bird_attrpath+'/image_attribute_labels.txt', 'r') as f:
            # save the attributes for which is_present is true
            attr_labels = [line.split() for line in f.readlines() if line.split()[2] == '1']

        tempdf = pd.DataFrame(attr_labels, columns='image_id attribute_id is_present certainty_id time trash'.split())
        cub_bbdf['image_id'] = cub_bbdf['image_id'].astype('str')
        attrdf = pd.merge(tempdf, cub_bbdf, on='image_id')

        attrdf['att'] = attrdf.attribute_id.apply(lambda x: attr_dict[x][0])
        attrdf['val'] = attrdf.attribute_id.apply(lambda x: attr_dict[x][1])
        attrdf['i_corpus'] = icorpus_code['cub_birds']

        column_order = 'i_corpus image_id att val'.split()
        cub_attrdf = attrdf[column_order]
        self._dumpDF(cub_attrdf, args.out_dir + '/cub_attrdf.json', args)

    # ======= CUB Birds 2011 Parts ========
    #
    def tsk_birdparts(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Caltech-UCSD Birds-200-2011 Parts', indent=4)

        bird_partpath = config.get('CUB_BIRDS', 'birds_base') + '/parts'

        with open(bird_partpath+'/part_locs.txt', 'r') as f:
            part_locs = [line.split() for line in f.readlines() if line.split()[4] == '1']

        bird_part_dict = {}
        with open(bird_partpath+'/parts.txt', 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ', 1)
                bird_part_dict[parts[0]] = parts[1]

        # this requires cub_bbdf to be present in the default out dir
        cub_bbdf = pd.read_json(args.out_dir + '/cub_bbdf.json.gz',
                                typ='frame', orient='split',
                                compression='gzip')

        tempdf = pd.DataFrame(part_locs, columns='image_id part_id x y visible'.split())
        cub_bbdf['image_id'] = cub_bbdf['image_id'].astype('str')
        partdf = pd.merge(tempdf, cub_bbdf, on='image_id')

        partdf['part_name'] = partdf.part_id.apply(lambda x: bird_part_dict[x])

        column_order = 'i_corpus image_id part_name x y'.split()
        cub_partdf = partdf[column_order]
        self._dumpDF(cub_partdf, args.out_dir + '/cub_partdf.json', args)

    # ======= CUB Birds 2011 Captions ========
    #
    def tsk_birdcap(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Caltech-UCSD Birds-200-2011 Captions', indent=4)

        bird_cappath = config.get('CUB_BIRDS', 'birds_caps')

        cub_bbdf = pd.read_json(args.out_dir + '/cub_bbdf.json.gz',
                                typ='frame', orient='split',
                                compression='gzip')

        caption_rows = []
        for path in glob.glob(bird_cappath+'/*/*.txt'):
            with open(path, 'r') as f:
                captions = [line for line in f.read().split('\n') if line != '']
            image_path = re.search(bird_cappath+'/(.*).txt', path).group(1)
            for cap in captions:
                caption_rows.append({'image_path': image_path+'.jpg',
                                     'caption': cap})

        captiondf = pd.DataFrame(caption_rows)
        completedf = pd.merge(captiondf, cub_bbdf, on='image_path')

        completedf['cat'] = completedf.image_path.apply(lambda x: re.search('(.*)/', x).group(1))

        column_order = 'i_corpus image_id caption cat'.split()
        cub_capdf = completedf[column_order]

        self._dumpDF(cub_capdf, args.out_dir + '/cub_capdf.json', args)

    # ======= ADE 20K part relations & objects========
    #
    def tsk_aderel(self):
        config = self.config
        args = self.args

        print_timestamped_message('...ADE 20K Part-of Relations & Objects', indent=4)

        ade_basepath = config.get('ADE_20K', 'ade_base')

        image_paths = ade_path_data(ade_basepath+'/index_ade20k.mat')

        corpus_id = icorpus_code['ade_20k']

        part_relations = []
        ade_objects = []
        for n, (image_cat, image_id, filename) in tqdm(enumerate(image_paths)):
            if 'outliers' not in image_cat and 'misc' not in image_cat:
                if 'training' in image_cat:
                    this_set = 'training'
                    this_cat = image_cat.split('training/')[1]
                elif 'validation' in image_cat:
                    this_set = 'validation'
                    this_cat = image_cat.split('validation/')[1]

                # print image_cat, image_id, filename
                seg_files = glob.glob(ade_basepath+'/'+image_cat+'/'+filename+'*.png')
                level_arrays = []
                for file in seg_files:
                    if 'seg' in file:
                        level_arrays.append((0, plt.imread(file)))
                    elif 'parts' in file:
                        level = re.search(r'.*parts_(.).png', file).group(1)
                        level_arrays.append((int(level), plt.imread(file)))
                level_arrays = sorted(level_arrays, key=itemgetter(0))
                level_masks = [(lvl, id_mask(array)) for lvl, array in level_arrays]

                # record the total number of objects for comparison with annotations
                object_no = 0
                for n, mask in level_masks:
                    object_no += len(np.unique(mask))-1  # not counting 0

                annotation_file = ade_annotation(ade_basepath, image_cat, filename)
                with open(annotation_file, 'r') as ann_f:
                    annotation_lines = ann_f.read().split('\n')
                annotation_lines = [ann for ann in annotation_lines if ann != '']

                # inconsistency check
                if len(annotation_lines) > object_no:
                    print(image_id, 'inc')
                    with open(args.out_dir + '/ade_inconsistent_images.txt', 'a') as f:
                        f.write(ade_basepath+image_cat+'/'+filename+'\n')
                    continue

                for level, mask in level_masks[1:]:
                    for small in np.unique(mask)[1:]:
                        small_mask = np.where(mask == small)
                        int_big = level_masks[int(level)-1][1]
                        big_mask = int_big[small_mask]
                        if all(big_mask == big_mask[0]):
                            part_relations.append({'i_corpus': corpus_id,
                                                   'image_id': image_id,
                                                   'region_id': big_mask[0],
                                                   'region_level': int(level)-1,
                                                   'part_id': small,
                                                   'part_level': int(level)})

                for this_line in annotation_lines:
                    this_line_split = this_line.split(' # ')
                    obj_id = this_line_split[0]
                    level = this_line_split[1]
                    wn_synset = this_line_split[3]
                    label = this_line_split[4]
                    if this_line_split[5] != "":
                        attrs = this_line_split[5].strip('\"')
                    else:
                        attrs = False
                    if this_line_split[2] == '0':
                        occl = False
                    else:
                        occl = True

                    bb = get_ade_bb(level_arrays[int(level)][1], obj_id)

                    # print obj_id, level, bb, image_id, label, wn_lemmas
                    ade_objects.append({'i_corpus': corpus_id,
                                        'image_id': image_id,
                                        'region_id': obj_id,
                                        'level': level,
                                        'label': label,
                                        'synset': wn_synset,
                                        'attr': attrs,
                                        'occl': occl,
                                        'bb': bb,
                                        'filename': filename+'.jpg',
                                        'image_cat': this_cat,
                                        'split': this_set
                                        })

        rel_columns = 'i_corpus image_id region_id region_level part_id part_level'.split()
        relations_df = pd.DataFrame(part_relations)
        relations_df = relations_df[rel_columns]
        self._dumpDF(relations_df, args.out_dir + '/ade_reldf.json', args)

        obj_columns = 'i_corpus image_id region_id level label synset attr occl bb image_cat split filename'.split()
        objects_df = pd.DataFrame(ade_objects)
        objects_df = objects_df[obj_columns]
        self._dumpDF(objects_df, args.out_dir + '/ade_objdf.json', args)

    # ======= ADE 20K images ========
    #
    def tsk_adeimgs(self):
        config = self.config
        args = self.args

        print_timestamped_message('...ADE 20K Image Dataframe', indent=4)

        ade_basepath = config.get('ADE_20K', 'ade_base')

        image_paths = ade_path_data(ade_basepath+'/index_ade20k.mat')
        corpus_id = icorpus_code['ade_20k']

        image_dataframe = []
        for (image_cat, image_id, filename) in image_paths:
            if 'outliers' not in image_cat and 'misc' not in image_cat:
                if 'training' in image_cat:
                    this_set = 'training'
                    this_cat = image_cat.split('training/')[1]
                elif 'validation' in image_cat:
                    this_set = 'validation'
                    this_cat = image_cat.split('validation/')[1]
                image_dataframe.append({'i_corpus': corpus_id,
                                        'image_id': image_id,
                                        'filename': filename+'.jpg',
                                        'image_cat': this_cat,
                                        'split': this_set})

        images_df = pd.DataFrame(image_dataframe)
        images_df = images_df['i_corpus image_id image_cat split filename'.split()]
        self._dumpDF(images_df, args.out_dir + '/ade_imgdf.json', args)

    # ======= Guess What? ========
    #
    def tsk_guesswhat(self):
        config = self.config

        corpus_id = icorpus_code['mscoco']

        print_timestamped_message('... Guess What dialogues', indent=4)

        gw_basepath = config.get('GUESSWHAT', 'gw_base')

        out = []

        for split_json_path in glob.glob(gw_basepath + '/*'):
            this_file = os.path.basename(split_json_path)
            this_split = this_file.split('.')[1]

            with gzip.open(split_json_path, 'r') as f:
                for n, this_line in enumerate(f.readlines()):
                    this_dial = json.loads(this_line)
                    # corpus_id, image_id, dialogue_id, turn_id, q, a, target, all_obj, success
                    image_id = this_dial['image']['id']
                    dialogue_id = n
                    target_obj = this_dial['object_id']
                    success = True if this_dial['status'] == 'success' else False
                    all_objs = [this_obj['id'] for this_obj in this_dial['objects']]
                    for this_turn in this_dial['qas']:
                        out.append((corpus_id,
                                    image_id,
                                    dialogue_id,
                                    this_turn['id'],
                                    this_turn['question'],
                                    this_turn['answer'],
                                    target_obj,
                                    all_objs,
                                    success,
                                    this_split))

        gw_df = pd.DataFrame(out,
                             columns='corpus_id image_id dial_id turn_id q a target all_objs success split'.split())
        self._dumpDF(gw_df, args.out_dir + '/gw_df.json', args)

    # ======= VisDial ========
    #
    def tsk_visdial(self):
        config = self.config

        corpus_id = icorpus_code['mscoco']

        print_timestamped_message('... VisDial dialogues', indent=4)

        vd_basepath = config.get('VISDIAL', 'vd_base')

        out = []

        for split_json_path in glob.glob(vd_basepath + '/*'):
            with gzip.open(split_json_path, 'r') as f:
                dataset = json.load(f)
                split = dataset['split']

                for n, this_dial in enumerate(dataset['data']['dialogs']):
                    image_id = this_dial['image_id']
                    dial_id = n
                    trigger_caption = this_dial['caption']
                    for m, this_turn in enumerate(this_dial['dialog']):
                        question = dataset['data']['questions'][this_turn['question']]
                        answer = dataset['data']['answers'][this_turn['answer']]
                        turn_id = m

                        out.append((corpus_id, image_id, dial_id, turn_id,
                                    question, answer, trigger_caption, split))
        vd_df = pd.DataFrame(out,
                             columns='corpus_id image_id dial_id turn_id question answer trigger_caption split'.split())

        self._dumpDF(vd_df, args.out_dir + '/vd_df.json', args)

    # ======= VQA ========
    #
    def tsk_vqa(self):
        config = self.config
        vqa_basepath = config.get('VQA', 'vqa_base')

        corpus_id = icorpus_code['mscoco']

        print_timestamped_message('... vqa', indent=4)

        out = []

        for split in ['train', 'val']:
            anno_path = 'v2_mscoco_{}2014_annotations.json'.format(split)
            ques_path = 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(split)
            with open(vqa_basepath + '/' + anno_path, 'r') as f:
                annos = json.load(f)
            with open(vqa_basepath + '/' + ques_path, 'r') as f:
                qs = json.load(f)

            qs = {q['question_id']: (q['image_id'], q['question'])
                  for q in qs['questions']}

            for answer in annos['annotations']:
                qii, q = qs[answer['question_id']]
                assert answer['image_id'] == qii, 'dataset inconsistent!'
                out.append((
                    corpus_id,
                    answer['image_id'],
                    answer['question_id'],
                    q,
                    answer['multiple_choice_answer'],
                    answer['question_type'],
                    split
                ))

        vqa_df = pd.DataFrame(out, columns='i_corpus image_id q_id q a q_type split'.split())
        self._dumpDF(vqa_df, args.out_dir + '/vqa.json', args)

    # ======= COCO Entities ========
    #
    def tsk_cocoent(self):
        config = self.config
        cocoent_path = config.get('COCOENT', 'cocoent')

        corpus_id = icorpus_code['mscoco']

        print_timestamped_message('... coco entities', indent=4)

        with open(cocoent_path, 'r') as f:
            ce = json.load(f)

        cap_df = []
        obj_df = []
        for this_image, this_image_dict in tqdm(ce.items()):
            detections = {}
            for this_cap, this_anno in this_image_dict.items():
                detections.update(this_anno['detections'])
                cap_df.append((corpus_id, this_image,
                               this_anno['split'],
                               this_cap,
                               serialise_cococap(this_cap, this_anno)))
            for ent_type, bbs in detections.items():
                for rid, bb in bbs:
                    bx, by = bb[0], bb[1]
                    bw, bh = bb[2] - bx, bb[3] - by
                    obj_df.append((corpus_id,
                                   this_image,
                                   rid,
                                   this_anno['split'],
                                   [bx, by, bw, bh],
                                   ent_type))

        cap_df = pd.DataFrame(cap_df, columns='i_corpus image_id split cap cap_ent'.split())
        obj_df = pd.DataFrame(obj_df, columns='i_corpus image_id region_id split bb type'.split())
        self._dumpDF(cap_df, args.out_dir + '/cocoent_capdf.json', args)
        self._dumpDF(obj_df, args.out_dir + '/cocoent_objdf.json', args)


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
                                 'grex', 'saiaprbb', 'mscocobb',
                                 'mscococap', 'mscococats',
                                 'grexbb', 'visgenimg', 'visgenreg',
                                 'visgenrel', 'visgenobj', 'visgenatt',
                                 'visgenvqa', 'visgenpar',
                                 'flickrbb', 'flickrcap', 'flickrobj',
                                 'birdbb', 'birdattr', 'birdparts', 'birdcap',
                                 'aderel', 'adeimgs',
                                 'guesswhat', 'visdial', 'vqa', 'cocoent',
                                 'all'],
                        help='''
                        task(s) to do. Choose one or more.
                        'all' runs all tasks.''')

    targs = parser.parse_known_args()
    args, unparsed_args = targs

    print("treated as task-specific parameters: ", unparsed_args)

    config = configparser.ConfigParser()

    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
    except IOError:
        print('no config file found at %s' % (args.config_file))
        sys.exit(1)

    tfs = TaskFunctions(targs, config)

    # FIXME: must be specified manually, unfortunately, as there
    #  are dependencies between the tasks.
    if 'all' in args.task:
        available_tasks = [this_method.replace('tsk_', '')
                           for this_method in dir(tfs)
                           if this_method.startswith('tsk_')]
        print('I will run all of:', available_tasks)
        args.task = available_tasks

    print_timestamped_message('starting to preprocess...')

    for task in args.task:
        tfs.exec_task(task)

    print_timestamped_message('... done!')
