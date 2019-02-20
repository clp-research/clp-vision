import numpy as np
import scipy.io
import re
import matplotlib.pyplot as plt


def id_mask(image):
    ''' translates the array of a segmentation file
       into a mask with the actual object ids'''
    array = image[:,:,2]
    d = {idx: num for num, idx in enumerate(np.unique(array))}
    return np.vectorize(d.__getitem__)(array)


def ade_path_data(matpath):
    """given the path to the index_ade20k.mat annoation file,
       returns (path, image id, filename) of all images"""
    ade_anns = scipy.io.loadmat(matpath)['index'][0][0]
    filenames = np.squeeze(ade_anns[1])
    pathnames = np.squeeze(ade_anns[0])
    image_data = []
    for i in range(len(filenames)):
        image_path = str(filenames[i][0]).split('ADE20K_2016_07_26/')[1]
        filename = str(pathnames[i][0]).split('.jpg')[0]
        image_id = re.search(r'([0-9]+)', filename).group(1)
        image_data.append((image_path, image_id, filename))
    return image_data


def ade_annotation(impath, img_cat, image_name):
    basepath = impath+'/'+img_cat+'/'+image_name
    return basepath + '_atr.txt'


def get_ade_mask(impath, img_cat, image_name):
    basepath = impath+'/'+img_cat+'/'+image_name
    return basepath + '_seg.png'


def get_ade_mask_parts(impath, img_cat, image_name, level):
    basepath = impath+'/'+img_cat+'/'+image_name
    return basepath + '_parts_{}.png'.format(level)


def get_ade_bb(seg, region_id):
    B = seg[:,:,2]
    i = int(region_id)
    mask = (B == ((np.unique(B))[i]))*1
    x1, y1 = np.nonzero(mask)[1].min(), np.nonzero(mask)[0].min()
    x2, y2 = np.nonzero(mask)[1].max(), np.nonzero(mask)[0].max()
    return [x1, y1, x2-x1, y2-y1]
