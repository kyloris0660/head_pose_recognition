import os
import sys
import re
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

ROOT_DIR = '/Users/kyloris/Projects/Mask_RCNN'
sys.path.append(ROOT_DIR)

import mrcnn.model as modellib
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = '/Users/kyloris/Downloads/HeadPoseImageDatabase/photo/'
embedding_path = '/Users/kyloris/Downloads/val2017/'

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)

file_names = next(os.walk(IMAGE_DIR))[2]
embedding_file_names = next(os.walk(embedding_path))[2]


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def embedding_pix(image, embedding_image, boxes, masks, class_ids):
    N = boxes.shape[0]
    # assert not N
    assert class_ids.shape[0] == masks.shape[-1]
    # masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if class_ids[i] != [1]:
            continue
        else:
            mask = masks[:, :, i]
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c],
                                          embedding_image[:, :, c])
    return image


def main():
    raw_data = os.listdir(IMAGE_DIR)
    enhance_data = os.listdir(embedding_path)
    vec = []
    num_enhance = 5
    vec_pix = np.zeros((2790 * num_enhance, 288, 384, 3))
    # vec_pix = np.zeros((5 * num_enhance, 288, 384, 3))
    cnt = 0
    for i in range(len(raw_data)):
        if os.path.splitext(IMAGE_DIR + raw_data[i])[-1] != '.jpg':
            pass
        else:
            image = img_to_array(Image.open(IMAGE_DIR + raw_data[i])).astype('uint8')
            a = re.findall(r'[-+]?\d+', raw_data[i])[-2:]
            a = [int(k) for k in a]
            r = model.detect([image], verbose=1)[0]
            for j in range(num_enhance):
                image_cp = image[:]
                enhance_image = img_to_array(Image.open(embedding_path + enhance_data[cnt])).astype(
                    'uint8')
                enhance_image = random_crop(enhance_image, (288, 384))
                vec_pix[cnt] = embedding_pix(image_cp, enhance_image, r['rois'], r['masks'], r['class_ids'])
                cnt += 1
                vec.append(a)
    vec = np.asarray(vec)

    assert vec.shape[0] == vec_pix.shape[0]

    np.save('ground_truth_enhanced.npy', vec)
    np.save('pix_data_enhanced.npy', vec_pix)


if __name__ == '__main__':
    main()
