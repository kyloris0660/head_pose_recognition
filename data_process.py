import os
import re
import numpy as np
from config import *
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split


def data2array():
    data = os.listdir(LABEL_PATH)
    vec = []
    vec_of_pix = np.zeros((2790, 288, 384, 3))
    for i in range(len(data)):
        if os.path.splitext(LABEL_PATH + data[i])[-1] != '.txt':
            pass
        else:
            with open (LABEL_PATH + data[i], 'r', encoding="latin-1") as f:
                foo = f.read()
                a = re.split(r'\n', foo)
                a = a[-5:-1]
                a = [int(i) for i in a]
                vec.append(a)
            photo = Image.open(DATA_PATH + data[i].strip('.txt') + '.jpg')
            vec_of_pix[i] = img_to_array(photo)
    vec = np.asarray(vec)
    np.save('ground_truth.npy', vec)
    np.save('pix_data.npy', vec_of_pix)


def gen_train_test(split_ratio=0.7, random_state=42):
    X = np.load('pix_data.npy')
    y = np.load('ground_truth.npy')

    assert X.shape[0] == y.shape[0]

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state)




