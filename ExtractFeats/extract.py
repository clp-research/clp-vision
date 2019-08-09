# coding: utf-8
'''
Use keras models to extract features from images, as specified
by bbdf (output of preproc.py)
'''

from __future__ import division

# import json
import argparse
import sys
import configparser
from os.path import isfile
from glob import glob

import pandas as pd
import numpy as np
import dask.array as da
import h5py

from tqdm import tqdm

from keras.models import Model
from keras import backend

sys.path.append('../Utils')
from utils import print_timestamped_message, code_icorpus, get_image_part
from utils import join_imagenet_id

backend.set_image_data_format('channels_last')


class FakeModel(object):
    def predict(self, X, n_feats=4096):
        return np.random.rand(X.shape[0], n_feats)


def compute_posfeats(img, bb):
    n_pos_feats = 7
    if bb is None:
        return np.zeros(n_pos_feats)

    ih, iw, _ = img.shape
    x, y, w, h = bb
    # x1, relative
    x1r = x / iw
    # y1, relative
    y1r = y / ih
    # x2, relative
    x2r = (x+w) / iw
    # y2, relative
    y2r = (y+h) / ih
    # area
    area = (w*h) / (iw*ih)
    # ratio image sides (= orientation)
    ratio = iw / ih
    # distance from center (normalised)
    cx = iw / 2
    cy = ih / 2
    bcx = x + w / 2
    bcy = y + h / 2
    distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)
    # done!
    return np.array([x1r, y1r, x2r, y2r, area, ratio, distance])


def compute_feats(config, args, bbdf, model, preproc,
                  xs=224, ys=224, batch_size=100):

    full_image = args.full_image

    filename = config.get('runtime', 'out_dir') + \
        '/%s_%s' % (
            config.get('runtime', 'this_bbdf'),
            config.get('runtime', 'model'))
    if full_image:
        filename += '-fi'
    # if isfile(filename + '.npz'):
    if len(glob(filename + '*')) != 0:
        print('Output for %s exists. Will not overwrite. ABORTING.' % (filename))
        return

    X_pos = []
    X_i = []
    ids = []
    file_counter = 1
    prev_iid, prev_img = (None, None)

    X_out = []
    write_flag = False
    write_count = 1
    minibatch_size = args.write_batch
    checkpts = minibatch_size
    
    # FIXME, for debugging only! Reduced size or starting with offset
    # bbdf = bbdf[:100]
    if args.bbdf_slice:
        s, e = [int(e) for e in args.bbdf_slice.split(':')]
        bbdf = bbdf[s:e]

    if len(bbdf) > args.max_singlefile:
        size_flag = True
    else:
        size_flag = False

    if full_image:
        bbdf = bbdf.drop_duplicates(subset='image_id')
        bbdf = bbdf.reset_index()

    # if 'region_id' in bbdf.columns:  # default
    reg_col = 'region_id'
    if 'obj_id' in bbdf.columns:  # some visgen bbdfs
        reg_col = 'obj_id'
    if 'subregion_id' in bbdf.columns:  # Flickr30k
        subreg = True
        subreg_column = 'subregion_id'
    elif 'level' in bbdf.columns:  # ADE20k
        subreg = True
        subreg_column = 'level'
    else:
        subreg = False

    for n, row in tqdm(bbdf.iterrows(), total=len(bbdf)):
        this_icorpus = row['i_corpus']
        this_image_id = row['image_id']

        if full_image:
            this_bb = None
            this_region_id = 0
        else:
            this_bb = row['bb']
            this_region_id = row[reg_col]

        if subreg:  # this means that we are reading in Flickr30k...
            # .. or ADE20k
            this_region_id = row[reg_col] + row[subreg_column] / 100

        #  When extracting feats for imagenet regions, must
        #  - create combined filename out of image_id and region_id
        #  - neutralise positional features, by setting bb given
        #    to pos feat computation to 0,0,w,h. So that all ImageNet
        #    regions end up with same positions.
        if code_icorpus[this_icorpus] == 'image_net':
            this_image_id_mod = join_imagenet_id(this_image_id,
                                                 this_region_id)
            this_bb_mod = [0, 0, this_bb[2], this_bb[3]]
        elif code_icorpus[this_icorpus] == 'ade_20k':
            # somewhat regrettably, ade20k wasn't preprocessed to
            # use our normal format. this is coming back to haunt
            # us here, as we need to create the image id from
            # other rows.. this will only work on ade_imgdf, not on ade_objdf
            this_image_id_mod = (row['split'], row['image_cat'], row['filename'])
            this_bb_mod = this_bb
        elif code_icorpus[this_icorpus] == 'cub_birds':
            this_image_id_mod = row['image_path']
            this_bb_mod = this_bb
        else:
            this_image_id_mod = this_image_id
            this_bb_mod = this_bb

        if this_bb_mod and np.min(this_bb_mod[2:]) <= 0:
            print('skipping over this image (%s,%d). Negative bb! %s' % \
                (code_icorpus[this_icorpus], this_image_id, str(this_bb_mod)))
            continue

        try:
            (prev_iid, prev_img), img_resized = \
            get_image_part(config, (prev_iid, prev_img),
                           this_icorpus, this_image_id_mod, this_bb_mod,
                           xs=xs, ys=ys)
        except ValueError as e:
            print('skipping over this image (%s,%d). corrupted??' % \
               (code_icorpus[this_icorpus], this_image_id))

        if len(prev_img.shape) != 3 or \
           (len(prev_img.shape) == 3 and prev_img.shape[2] != 3):
            print('skipping over this image (%s,%d). b/w?' % \
                (code_icorpus[this_icorpus], this_image_id))
            continue

        # If we continue below this line, getting region worked
        X_i.append(img_resized)
        this_pos_feats = compute_posfeats(prev_img, this_bb_mod)
        X_pos.append(this_pos_feats)
        ids.append(np.array([this_icorpus, this_image_id, this_region_id]))

        # is it time to do the actual extraction on this batch
        if (n+1) % batch_size == 0 or n+1 == len(bbdf):
            print_timestamped_message('new batch! (%d %d) Extracting!...' %
                                      (file_counter, n), indent=4)

            try:
                X_i = np.array(X_i)
                # print X_i.shape
                X = model.predict(preproc(X_i.astype('float64')))
            except ValueError:
                print('Exception! But why? Skipping this whole batch..')
                X_i = []
                ids = []
                X_pos = []
                continue
                # raise e

            X_ids = np.array(ids)
            X_pos = np.array(X_pos)
            print(X_ids.shape, X.shape, X_pos.shape)
            if full_image:
                X_out_buff = da.from_array(np.hstack([X_ids, X]), chunks=(1000, 1000))
                X_out.append(X_out_buff)
            else:
                X_out_buff = da.from_array(np.hstack([X_ids, X, X_pos]), chunks=(1000, 1000))
                X_out.append(X_out_buff)

            ids = []
            X_pos = []
            X_i = []
            file_counter += 1
    
        # testing out mini-batch extractions
        if n >= checkpts or n+1 == len(bbdf):
            write_flag = True
            checkpts += minibatch_size

        if write_flag and size_flag and (not args.dry_run or args.write_dummy):
            write_flag = False

            write_buffer = np.concatenate(X_out, axis=0)
            # np.savez_compressed(filename + "_" + str(write_count), write_buffer)
            # uncompressed hdf5, after all?
            outfilename = filename + "_" + str(write_count) + ".hdf5"
            with h5py.File(outfilename, 'w') as f:
                f.create_dataset('img_feats', data=write_buffer)

            # write_buffer = da.concatenate(X_out, axis=0)
            # da.to_hdf5(filename + "_" + str(write_count) + ".hdf5",
            #            'img_feats', write_buffer,
            #            compression="gzip", compression_opts=9,
            #            shuffle=True, chunks=True)

            write_count += 1
            X_out = []

    # and back to the for loop
    if not size_flag and (not args.dry_run or args.write_dummy):
        # X_out = da.concatenate(X_out, axis=0)
        X_out = np.concatenate(X_out, axis=0)

        print_timestamped_message('Made it through! Writing out..', indent=4)
        print(X_out.shape)

        with h5py.File(filename + '.hdf5', 'w') as f:
            f.create_dataset('img_feats', data=X_out)


# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use keras ConvNets to extract image features')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file specifying data paths.
                        default: '../Config/default.cfg' ''',
                        default='../Config/default.cfg')
    parser.add_argument('-o', '--out_dir',
                        help='''
                        where to put the resulting files.
                        default: './ExtractOut' ''')
    parser.add_argument('-b', '--bbdf_dir',
                        help='''
                        Where to look for the bbdf file.
                        default: '../Preproc/PreProcOut' ''')
    parser.add_argument('-s', '--size_batch',
                        help='''
                        How many images to give to model as one batch.
                        default: 100''',
                        type=int,
                        default=100)
    parser.add_argument('--write_batch',
                        help='''
                        Rows after which to write out a dask array .
                        default: 100000''',
                        type=int,
                        default=100000)
    parser.add_argument('--max_singlefile',
                        help='''
                        Max n rows single file.
                        default: 200000''',
                        type=int,
                        default=200000)
    parser.add_argument('-f', '--full_image',
                        action='store_true',
                        help='Extract whole image, ignore BBs.')

    parser.add_argument('-d', '--dry_run',
                        action='store_true',
                        help='Don\'t actually run the extraction model')
    parser.add_argument('--write_dummy',
                        action='store_true',
                        help='Write out (dummy) file even in dry_run')
    parser.add_argument('--bbdf_slice',
                        help='Slice of bbdf to extract, for debugging')

    parser.add_argument('model',
                        choices=['vgg19-fc2', 'rsn50-max'],
                        help='''
                        Which model/layer to use for extraction.''')
    parser.add_argument('bbdf',
                        nargs='+',
                        help='''
                        Which bddf(s) to run this on.''')
    args = parser.parse_args()

    config = configparser.ConfigParser()

    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
    except IOError:
        print('no config file found at %s' % (args.config_file))
        sys.exit(1)

    if args.bbdf_dir:
        bbdf_dir = args.bbdf_dir
    elif config.has_option('DSGV-PATHS', 'preproc_path'):
        bbdf_dir = config.get('DSGV-PATHS', 'preproc_path')
    else:
        bbdf_dir = '../Preproc/PreprocOut'

    if args.out_dir:
        out_dir = args.out_dir
    elif config.has_option('DSGV-PATHS', 'extract_out_dir'):
        out_dir = config.get('DSGV-PATHS', 'extract_out_dir')
    else:
        out_dir = './ExtractOut'

    config.add_section('runtime')
    config.set('runtime', 'out_dir', out_dir)

    print(bbdf_dir, out_dir)

    if args.full_image:
        print("Full Image Mode Selected! Extraction will take whole image as input.")

    # default dimensions
    xs, ys = 224, 224

    arch, layer = args.model.split('-')
    print(args.bbdf, arch, layer)
    config.set('runtime', 'model', args.model)

    if arch == 'vgg19' and not args.dry_run:
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input as preproc
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input,
                      outputs=base_model.get_layer(layer).output)
    if arch == 'rsn50' and not args.dry_run:
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input as preproc
        # base_model = ResNet50(weights='imagenet')
        model = ResNet50(weights='imagenet', include_top=False, pooling=layer)
        # model = Model(inputs=base_model.input,
        #               outputs=base_model.get_layer(layer).output)
    if args.dry_run:
        model = FakeModel()
        preproc = lambda x: x

    print_timestamped_message('starting to extract, using %s %s...' %
                              (arch, layer))

    for this_bbdf in args.bbdf:
        print_timestamped_message('... %s' % (this_bbdf), indent=4)
        this_bbdf_base = bbdf_dir + '/' + this_bbdf + '.json'
        if isfile(this_bbdf_base + '.gz'):
            this_bbdf_path = this_bbdf_base + '.gz'
            bbdf = pd.read_json(this_bbdf_path,
                                orient='split',
                                compression='gzip')
        else:
            this_bbdf_path = this_bbdf_base
            if not isfile(this_bbdf_base):
                print("bbdf file (%s) not found. Aborting." % (this_bbdf_path))
                sys.exit(1)
            bbdf = pd.read_json(this_bbdf_base,
                                orient='split')
        print(this_bbdf_path)

        config.set('runtime', 'this_bbdf', this_bbdf)

        compute_feats(config, args, bbdf, model, preproc,
                      xs=xs, ys=ys, batch_size=args.size_batch)
