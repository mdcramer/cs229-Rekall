'''
Detect objects using FAIR's maskrcnn-benchmark repo.

To run this code, you'll need to download the maskrcnn-benchmark code
(last tested on commit 24c8c90efdb7cc51381af5ce0205b23567c3cd21).
Follow the installation instructions in Install.md, and then put this scripts
in the demo folder.
'''

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import matplotlib.pyplot as plt
import pickle

ROOT_PATH = '/lfs/1/danfu/rekall_experiments/parking_space_experiments'
IMAGE_FOLDER = 'images_ffmpeg'
video_names = sorted(os.listdir(os.path.join(ROOT_PATH, IMAGE_FOLDER)))
BBOXES_FOLDER = 'bboxes'

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

print("Loaded model")

def load_image(path):
    img = Image.open(path).convert('RGB')
    
    return np.array(img)[:, :, [2, 1, 0]]

for video in video_names:
    print(video)
    images_path = os.path.join(os.path.join(ROOT_PATH, IMAGE_FOLDER), video)
    
    images = sorted(os.listdir(images_path))
    
    all_bboxes = []
    for image_name in tqdm(images):
        image_path = os.path.join(images_path, image_name)
        img = load_image(image_path)
        bboxes = coco_demo.compute_prediction(img)
        top_bboxes = coco_demo.select_top_predictions(bboxes)
        labels = top_bboxes.get_field("labels").tolist()
        labels = [coco_demo.CATEGORIES[i] for i in labels]
        scores = top_bboxes.get_field("scores").tolist()
        
        all_bboxes.append([
            bbox + [label] + [score] + [image_name]
            for bbox, label, score in zip(
                top_bboxes.bbox.tolist(),
                labels,
                scores
            )
        ])
        
    bboxes_path = os.path.join(
        ROOT_PATH, os.path.join(BBOXES_FOLDER, video))
    os.makedirs(bboxes_path, exist_ok=True)
    output_path = os.path.join(bboxes_path, 'bboxes.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_bboxes, f)