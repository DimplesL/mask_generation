
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import h5py
import coco
import utils
import model as modellib
import visualize

# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 20
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512
    BATCH_SIZE = 20
config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
video_image_path = '/hdfs/qiuyurui/DATA/youtube_image_5/'   #youtube_image/'
video_all = {}
all_image_path = []
name_with_path = []
for video_forder in os.listdir(video_image_path):
    video_name = video_forder
    hdf5_p = '/hdfs/qiuyurui/DATA/hdf5_msvd/'
    hdf5_path = os.path.join(hdf5_p, video_forder)
    if not os.path.exists(hdf5_path):
        os.mkdir(hdf5_path)
    subdir = os.path.join(video_image_path, video_name)
    image_all = {}
    for image_file in os.listdir(subdir):
        image_name = image_file[:-4]
        image_path = os.path.join(subdir, image_file)
        all_image_path.append(image_path)
        image_all[image_name] = image_path
        name_with_path.append({'video':video_name,
                          'image_name':image_name,'image_path':image_path})
    video_all[video_name] = image_all
#         print(image_name)
#     for 
# print(all_image_path[100])
# print(video_all['vid1']['0001'])
print(len(name_with_path))

def load_image_for_batch(image_path, config):
    image = skimage.io.imread(image_path)
#     shape = image.shape
    image, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    return image

#def generate_mask_batch(batch_size = config.BATCH_SIZE, config, name_with_path):
number_image = len(name_with_path)
last_number = number_image


#     for n in rang(batch_num):
n = 0
batch_size = config.BATCH_SIZE
batch_num = number_image/batch_size
images_batchs = np.zeros((batch_size, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3))
m = 1
# last_number = last_number - batch_size
while last_number >= batch_size:
    vid_image = []
    r = []
    for i in range(batch_size):
        image_path = name_with_path[n]['image_path']
        video = name_with_path[n]['video']
        image_name = name_with_path[n]['image_name']
        resize_image = load_image_for_batch(image_path, config)
        images_batchs[i] = resize_image.copy()
        vid_image.append({'video':video, 'image_name': image_name})
        n = n+1
    last_number = last_number - batch_size
    print(len(images_batchs))
    r = model.detect(images_batchs)
    for i in range(batch_size):
        f = h5py.File('/hdfs/qiuyurui/DATA/hdf5_msvd/{}/{}.h5'.format(vid_image[i]['video'],vid_image[i]['image_name']),'w')
        f['features'] = r[i]['features']
        f['masks'] = r[i]['masks']
        f['class_ids'] = r[i]['class_ids']
        f['rois'] = r[i]['rois']
        f.close()
    print(m,' batch number / {}'.format(batch_num))
    m = m+1 
    
#     break
if last_number > 0 and last_number < batch_size:
    config.BATCH_SIZE = last_number
    config.IMAGES_PER_GPU = last_number
    batch_size = last_number
    images_batchs = np.zeros((batch_size, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3))
    vid_image = []
    r = []
    for i in range(batch_size):
        image_path = name_with_path[n]['image_path']
        video = name_with_path[n]['video']
        image_name = name_with_path[n]['image_name']
        resize_image = load_image_for_batch(image_path, config)
        images_batchs[i] = resize_image.copy()
        vid_image.append({'video':video, 'image_name': image_name})
        n = n+1
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    r = model.detect(images_batchs)
    for i in range(batch_size):
        f = h5py.File('/hdfs/qiuyurui/DATA/hdf5_msvd/{}/{}.h5'.format(vid_image[i]['video'],vid_image[i]['image_name']),'w')
        f['features'] = r[i]['features']
        f['masks'] = r[i]['masks']
        f['class_ids'] = r[i]['class_ids']
        f['rois'] = r[i]['rois']
        f.close()
    print(m,' batch number / {}'.format(batch_num))
#     print(len(r))
