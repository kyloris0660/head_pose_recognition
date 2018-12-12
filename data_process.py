import os
import re
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img


data_path = '/Users/kyloris/Downloads/HeadPoseImageDatabase/txt/'
data = os.listdir(data_path)
vec = []
vec_of_pix = np.zeros((2790, 288, 384, 3))
for i in range(len(data)):
    if os.path.splitext(data_path + data[i])[-1] != '.txt':
        pass
    else:
        with open (data_path + data[i], 'r', encoding="latin-1") as f:
            foo = f.read()
            a = re.split(r'\n', foo)
            a = a[-5:-1]
            a = [int(i) for i in a]
            vec.append(a)
        photo = Image.open('/Users/kyloris/Downloads/HeadPoseImageDatabase/photo/' + data[i].strip('.txt') + '.jpg')
        vec_of_pix[i] = img_to_array(photo)
vec = np.asarray(vec)
np.save('ground_truth.npy', vec)
np.save('pix_data.npy', vec_of_pix)



