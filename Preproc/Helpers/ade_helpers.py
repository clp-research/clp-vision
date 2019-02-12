import numpy as np
import scipy.io
import re

def id_mask(image):
    ''' translates the array of a segmentation file into a mask with the actual object ids'''
    array = image[:,:,2]
    d = {num:n for n, num in enumerate(np.unique(array))}
    return np.vectorize(d.__getitem__)(array)

def ade_path_data(matpath):
    """given the path to the index_ade20k.mat annoation file, returns (path, image id, filename) of all images"""
    ade_anns = scipy.io.loadmat(matpath)['index'][0][0]
    filenames = np.squeeze(ade_anns[1])
    pathnames = np.squeeze(ade_anns[0])
    image_data = []
    for i in range(len(filenames)):
        image_path = str(filenames[i][0])
        filename = str(pathnames[i][0]).split('.jpg')[0]
        image_id = re.search(r'([0-9]+)', filenames[i][0]).group(1)
        image_data.append((image_path, image_id, filename))
    return image_data
