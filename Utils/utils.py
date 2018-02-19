# coding: utf-8
from __future__ import division

import scipy.io
import numpy as np

icorpus_code = {
    'saiapr': 0,           # the original SAIAPR corpus; original regions
    'mscoco': 1,           # MSCOCO; original bounding boxes
    'saiapr_berkeley': 2,  # SAIAPR, with berkeley region proposals
    'mscoco_grprops': 3,   # MSCOCO, region proposals as per google refexp
    'image_net': 4,        # ImageNet; with bbs
    'visual_genome': 5,    # VG, images and regions from visual genome
    'ade_20k': 6,          # ADE, images from ade 20k corpus
    'tingdataset': 7  # https://tingh.github.io/resources/object_description
    }

code_icorpus = {item: key for key, item in icorpus_code.items()}


def saiapr_basepath(image_id):
    '''return the basepath for an SAIAPR image, given the image ID'''
    if len(str(image_id)) == 5:
        directory = str(image_id)[:2]
    elif len(str(image_id)) == 4:
        directory = '0' + str(image_id)[0]
    else:
        directory = '00'
    return directory


def saiapr_image_filename(config, image_id):
    '''return the path of a SAIAPR image, given the image ID'''
    directory = saiapr_basepath(image_id)
    return config.get('SAIAPR', 'saiapr_img_base') + '/' + \
        directory + '/images/' + str(image_id) + '.jpg'


def saiapr_mask_filename(config, image_id, region_id):
    '''return the path of a SAIAPR mask, given the image ID and region ID'''
    directory = saiapr_basepath(image_id)
    return config.get('SAIAPR', 'saiapr_img_base') +\
        '/' + directory + '/segmentation_masks/' + \
        str(image_id) + '_' + str(region_id) + '.mat'


def get_saiapr_bb(config, image_id, region_id):
    '''get the bounding box of an SAIAPR region, given image and region IDs'''
    mask_path = saiapr_mask_filename(config, image_id, region_id)
    # print mask_path
    mask = scipy.io.loadmat(mask_path)
    mask = mask['segimg_t']
    mask = mask + 1
    x1, y1 = np.nonzero(mask)[1].min(), np.nonzero(mask)[0].min()
    x2, y2 = np.nonzero(mask)[1].max(), np.nonzero(mask)[0].max()
    return [x1, y1, x2-x1, y2-y1]
