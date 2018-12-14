import os
import re
import numpy as np
from config import *
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split


def data2array():
    data = os.listdir(DATA_PATH)
    vec = []
    vec_of_pix = np.zeros((2790, 288, 384, 3))
    cnt = 0
    for i in range(len(data)):
        if os.path.splitext(DATA_PATH + data[i])[-1] != '.jpg':
            pass
        else:
            a = re.findall(r'[-+]?\d+', data[i])[-2:]
            a = [int(i) for i in a]
            vec.append(a)
            
            photo = Image.open(DATA_PATH + data[i])
            vec_of_pix[cnt] = img_to_array(photo)
            cnt += 1
    vec = np.asarray(vec)

    assert vec.shape[0] == vec_of_pix.shape[0] == 2790

    np.save('ground_truth.npy', vec)
    np.save('pix_data.npy', vec_of_pix)


def gen_train_test(split_ratio=0.7, random_state=42):
    X = np.load('pix_data.npy')
    y = np.load('ground_truth.npy')

    assert X.shape[0] == y.shape[0]

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state)


if '__name__' == '__main__':
    data2array()




