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
from ijson import items
import cPickle as pickle
import logging
from itertools import chain

import numpy as np
import scipy.io
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

N_VISGEN_IMG = 108077
#  The number of images in the visgen set, for the progress bar


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

        all_coco_files = [e for e in all_coco_files if type(e) is int]

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

    # ======= Visual Genome Relationships ========
    #
    def tsk_visgenrel(self):
        config = self.config
        args = self.args

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
                                out.append((this_rel,
                                            rel['predicate'],
                                            rel['relationship_id'],
                                            this_sub,
                                            rel['subject']['object_id'],
                                            this_obj,
                                            rel['object']['object_id'],
                                            image_id))
        vgrel_df = pd.DataFrame(out,
                                columns='rel_syn predicate rel_id sub_syn sub_id obj_syn obj_id image_id'.split())
        self._dumpDF(vgrel_df, args.out_dir + '/vgreldf.json', args)

    # ======= Visual Genome Objects ========
    #
    def tsk_visgenobj(self):
        config = self.config
        args = self.args

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
                            out.append((obj['object_id'],
                                        image_id,
                                        this_syn,
                                        this_name,
                                        (obj['x'], obj['y'], obj['w'], obj['h'])))
        vgobj_df = pd.DataFrame(out,
                                columns='obj_id image_id syn name bb'.split())
        self._dumpDF(vgobj_df, args.out_dir + '/vgobjdf.json', args)

    # ======= Visual Genome Attributes ========
    #
    def tsk_visgenatt(self):
        config = self.config
        args = self.args

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
                    out.append((obj['object_id'],
                                image_id,
                                atts))
        vgatt_df = pd.DataFrame(out,
                                columns='obj_id image_id attributes'.split())
        self._dumpDF(vgatt_df, args.out_dir + '/vgattdf.json', args)

    # ======= Visual Genome VQAs ========
    #
    def tsk_visgenvqa(self):
        config = self.config
        args = self.args

        print_timestamped_message('... VisualGenome VQAs', indent=4)

        vgvqa_path = config.get('VISGEN', 'visgen_12') + '/jsons/question_answers.json'
        with open(vgvqa_path, 'r') as f:
            out = []
            iterator = items(f, 'item')
            for n, entry in enumerate(tqdm(iterator, total=N_VISGEN_IMG)):
                for this_qa in entry['qas']:
                    out.append((this_qa['image_id'],
                                this_qa['qa_id'],
                                this_qa['question'],
                                this_qa['answer'],
                                this_qa['q_objects'],
                                this_qa['a_objects']))

        vgvqa_df = pd.DataFrame(out,
                                columns='image_id qa_id q a q_objs a_objs'.split())
        visgenqamap_p = config.get('VISGEN', 'visgen_12') + '/jsons/qa_to_region_mapping.json'

        with open(visgenqamap_p, 'r') as f:
            vgqamap = json.load(f)

        vgqamap = dict([(int(e[0]), e[1]) for e in vgqamap.items()])

        vgqamap_df = pd.DataFrame(vgqamap.items(), columns='qa_id region_id'.split())

        vgvqa_df = vgvqa_df.merge(vgqamap_df, how='left')

        self._dumpDF(vgvqa_df, args.out_dir + '/vgvqadf.json', args)

    # ======= Flickr 30k Entities RefDf ========
    #
    def tsk_flickrref(self):
        config = self.config
        args = self.args

        print_timestamped_message('... Flickr 30k Entities Refering Expressions', indent=4)

        flckrann_path = config.get('FLICKR', 'flickr_annotations')
        data = []

        for filename in os.listdir(flckrann_path):
            sents = []
            with open('..\Data\Annotations\Flickr30kEntities\Sentences\854749.txt', 'r') as f:
                for line in f:
                    sents.append(line)

            this_id = filename.split('.')[0]

            for sentence in sents:
                row = {}
                row['image_id'] = this_id
                row['caption_annotated'] = sentence
    
                temp = re.sub('[\]\[\n]','',sentence)
                row['caption_raw'] = ' '.join([word.lower() for word in temp.split() if not word.startswith('/')])
            
                these_entities = []
                for entity in re.findall('\[.*?\]',sentence):
                    these_entities.append(re.search('#(.*)/',entity).group(1))
                row['entities'] = these_entities
                data.append(row)
        flickr_refdf = pd.DataFrame(data)

        print len(flickr_refdf)

        self._dumpDF(flickr_refdf, args.out_dir + '/flickr_refdf.json', args)

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
                                 'grexbb', 'visgenreg', 'visgenrel',
                                 'visgenobj', 'visgenatt', 'visgenvqa',
                                 'flickrbb', 'flickrref',
                                 'all'],
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
