# coding: utf-8
'''
Pre-compute similarity matrices for images.

Currently implemented: Similarity based on annotation of objects present,
for MSCOCO and visual genome.


TODO:
x Looks like I can't compute similarity between all ~100k visgen
  images; process gets killed. (Memory?). Should subsample to some
  smaller number that works? Or do batches within which similarity is
  computed? (So that for each image there is a set of distractors.)
x Also, throw away columns?.. Ah, no.. Well. Could precompile a different
  format. For each image, only keep the n most similar... Instead of keeping
  the actual similarity matrix!
'''

from __future__ import division

import sys
import ConfigParser
import argparse
import codecs
from collections import Counter, defaultdict
from os.path import isfile

from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import numpy as np

sys.path.append('../Utils')
from utils import print_timestamped_message, code_icorpus

MAX_ROWS = 30000  # max images to compare. beyond it gets too slow.


def load_imsim(path):
    npl = np.load(path)
    return npl['arr_0'], dict(npl['arr_1'].tolist())


def n_most_sim(obsims, row2imid, target_row, n=3):
    '''
    Utility function for looking up a similarity ranking in the
    right batch and row, and then translating the batch column
    (similar image) into original row number so that the image id
    can be looked up in row2imid.
    '''
    batch_size = len(obsims[0])
    in_batch = target_row // batch_size
    translated_row = target_row - (batch_size * in_batch)
    this_row = obsims[in_batch][translated_row][0:n+1]
    return [row2imid[r + (batch_size * in_batch)] for r in this_row]


def get_sim_mat(df, id_col='image_id', cat_col='cat',
                n_dims=False, max_row=MAX_ROWS, n_most=200):
    print_timestamped_message('... compiling objects per image list')
    im2ob = defaultdict(list)
    df.apply(lambda row: im2ob[row[id_col]].append(row[cat_col]), axis=1)
    im2ob = {key: Counter(val) for key, val in im2ob.items()}
    row2imid = {n: iid for n, iid in enumerate(im2ob.keys())}

    print_timestamped_message('... vectorizing')
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(im2ob.values())

    if n_dims:
        print_timestamped_message('... reducing dimensionality')
        svd = TruncatedSVD(n_components=n_dims, n_iter=7, random_state=42)
        X = svd.fit_transform(X)

    print_timestamped_message('... and finally, computing similarities')

    objsim_most_sim_out = []
    for i in range(0, len(X), max_row):
        print_timestamped_message('... ... batch %d' % (i/max_row+1))
        objsim_pdist = pdist(X[i:i+max_row], 'cosine')
        objsim = squareform(objsim_pdist)
        objsim_most_sim = np.apply_along_axis(lambda x: np.argsort(x)[:n_most], 1, objsim)
        objsim_most_sim_out.append(objsim_most_sim)

    return objsim_most_sim_out, row2imid


# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use bbdfs to compute similarity btw images')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file specifying data paths.
                        default: '../Config/default.cfg' ''',
                        default='../Config/default.cfg')
    parser.add_argument('-b', '--bbdf_dir',
                        help='''
                        Where to look for the bbdf file.
                        default: '../Preproc/PreProcOut' ''')
    parser.add_argument('corp',
                        nargs='+',
                        choices=['mscoco', 'visgen'],
                        help='''
                        Which corpus / corpora to run this on.''')
    args = parser.parse_args()

    config = ConfigParser.SafeConfigParser()

    try:
        with codecs.open(args.config_file, 'r', encoding='utf-8') as f:
            config.readfp(f)
    except IOError:
        print 'no config file found at %s' % (args.config_file)
        sys.exit(1)

    if args.bbdf_dir:
        bbdf_dir = args.bbdf_dir
    elif config.has_option('DSGV-PATHS', 'bbdf_dir'):
        bbdf_dir = config.get('DSGV-PATHS', 'bbdf_dir')
    else:
        bbdf_dir = '../Preproc/PreprocOut'

    for this_corp in args.corp:

        if this_corp == 'mscoco':
            print_timestamped_message('Computing sims for MSCOCO')

            outfilename = bbdf_dir + '/mscoco_sim'
            if isfile(outfilename + '.npz'):
                print '%s exists. Will not overwrite. ABORTING.' % (outfilename + '.npz')
                break

            print_timestamped_message('... loading up bbdf')
            mscoco_bbdf = pd.read_json(bbdf_dir + '/mscoco_bbdf.json.gz',
                                       typ='frame',
                                       orient='split', compression='gzip')

            sim_sq, row2imid = get_sim_mat(mscoco_bbdf)

            print_timestamped_message('... compressing and writing to disk')
            np.savez_compressed(outfilename, sim_sq, row2imid)

        if this_corp == 'visgen':
            print_timestamped_message('Computing sims for Visual Genome')

            outfilename = bbdf_dir + '/visgen_sim'
            if isfile(outfilename + '.npz'):
                print '%s exists. Will not overwrite. ABORTING.' % (outfilename + '.npz')
                break

            print_timestamped_message('... loading up bbdf')
            visgen_objdf = pd.read_json(bbdf_dir + '/vgobjdf.json.gz',
                                        typ='frame', orient='split',
                                        compression='gzip')
            visgen_objdf = visgen_objdf[~visgen_objdf['obj_id'].duplicated()]
            visgen_objdf = visgen_objdf[~visgen_objdf['syn'].isnull()]

            sim_sq, row2imid = get_sim_mat(visgen_objdf, cat_col='syn',
                                           n_dims=50, max_row=MAX_ROWS)

            print_timestamped_message('... compressing and writing to disk')
            np.savez_compressed(outfilename, sim_sq, row2imid)

    print_timestamped_message('Done!')
