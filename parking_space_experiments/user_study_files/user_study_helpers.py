from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat
from vgrid_jupyter import VGridWidget
import urllib3, requests, os
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Hack to disable warnings about olimar's certificate
urllib3.disable_warnings()

VIDEO_COLLECTION_BASEURL = "https://olimar.stanford.edu/hdd/parking_lot/user_study"
VIDEO_METADATA_FILENAME = 'metadata.json'

# Load video file metadata
video_metadata = [ VideoMetadata(v['filename'], id=v['id'], fps=v['fps'],
                                 num_frames=v['num_frames'], width=v['width'],
                                 height=v['height'])
                  for v in requests.get(os.path.join(
                      VIDEO_COLLECTION_BASEURL, VIDEO_METADATA_FILENAME),
                                        verify=False).json() ]

VIDEO_FOLDER = 'videos'
BBOX_FOLDER = 'bboxes'
GT_FOLDER = 'empty_spaces'

dev_set = requests.get(
    os.path.join(VIDEO_COLLECTION_BASEURL, 'dev.txt'), verify=False
).content.decode('utf-8').strip().split('\n')
test_set = requests.get(
    os.path.join(VIDEO_COLLECTION_BASEURL, 'test.txt'), verify=False
).content.decode('utf-8').strip().split('\n')

video_metadata_dev = [
    vm
    for vm in video_metadata if vm.path in dev_set
]
video_metadata_test = [
    vm
    for vm in video_metadata if vm.path in test_set
]

def get_bboxes(dataset):
    interval = 30
    bboxes = [
        pickle.loads(requests.get(
            os.path.join(
                os.path.join(VIDEO_COLLECTION_BASEURL, BBOX_FOLDER),
                os.path.join(vm.path[:-4], 'bboxes.pkl')
            ),
            verify=False
        ).content)
        for vm in (video_metadata_dev if dataset == 'dev' else video_metadata_test)
    ]
    bboxes_ism = IntervalSetMapping({
        metadata.id: IntervalSet([
            Interval(
                Bounds3D(
                    t1 = 30 * i / metadata.fps,
                    t2 = 30 * (i + interval) / metadata.fps,
                    x1 = bbox[0] / metadata.width,
                    x2 = bbox[2] / metadata.width,
                    y1 = bbox[1] / metadata.height,
                    y2 = bbox[3] / metadata.height
                ),
                payload = { 'class': bbox[4], 'score': bbox[5] }
            )
            for i, frame in enumerate(bbox_frame_list) if (i % interval == 0)
            for bbox in frame
        ])
        for bbox_frame_list, metadata in tqdm(
            zip(bboxes,
                (video_metadata_dev if dataset == 'dev' else video_metadata_test)),
            total = len(bboxes))
    })
    
    return bboxes_ism

def get_gt(dataset='dev'):
    interval = 30
    empty_parking_spaces = [
        pickle.loads(requests.get(
            os.path.join(
                os.path.join(VIDEO_COLLECTION_BASEURL, GT_FOLDER),
                os.path.join(vm.path[:-4], 'gt.pkl')
            ),
            verify=False
        ).content)
        for vm in (video_metadata_dev if dataset == 'dev' else video_metadata_test)
    ]
    gt_ism = IntervalSetMapping({
        metadata.id: IntervalSet([
            Interval(
                Bounds3D(
                    t1 = 30 * i / metadata.fps,
                    t2 = 30 * (i + interval) / metadata.fps,
                    x1 = bbox[0] / metadata.width,
                    x2 = bbox[2] / metadata.width,
                    y1 = bbox[1] / metadata.height,
                    y2 = bbox[3] / metadata.height
                )
            )
            for i, frame in enumerate(space_frame_list) if (i % interval == 0)
            for bbox in frame
        ])
        for space_frame_list, metadata in tqdm(
            zip(empty_parking_spaces,
                (video_metadata_dev if dataset == 'dev' else video_metadata_test)),
            total = len(empty_parking_spaces))
    })
    
    return gt_ism

def visualize_boxes(box_list):
    vgrid_spec = VGridSpec(
        video_meta = video_metadata,
        vis_format = VideoBlockFormat(imaps = [
            (str(i), box)
            for i, box in enumerate(box_list)
        ]),
        video_endpoint = os.path.join(VIDEO_COLLECTION_BASEURL, VIDEO_FOLDER)
    )
    return VGridWidget(vgrid_spec = vgrid_spec.to_json_compressed())

def compute_ap(predictions, gt):
    from sklearn.metrics import average_precision_score
    import numpy as np
    
    true_positives = predictions.filter_against(
        gt,
        predicate = and_pred(
            Bounds3D.T(equal()),
            iou_at_least(0.5)
        ),
        window = 0.0,
        progress_bar = True
    )
    false_positives = predictions.minus(
        true_positives,
        predicate = and_pred(
            Bounds3D.T(equal()),
            iou_at_least(0.5)
        ),
        window = 0.0,
        progress_bar = True
    )
    false_negatives = gt.minus(
        predictions,
        predicate = and_pred(
            Bounds3D.T(equal()),
            iou_at_least(0.5)
        ),
        window = 0.0,
        progress_bar = True
    )
    
    tp_count = sum(true_positives.size().values())
    fp_count = sum(false_positives.size().values())
    fn_count = sum(false_negatives.size().values())
    
    y_true = np.concatenate([
        np.ones(tp_count),
        np.zeros(fp_count),
        np.zeros(1000 * sum(gt.size().values())),
        np.ones(fn_count),
    ])
    y_scores = np.concatenate([
        np.ones(tp_count),
        np.ones(fp_count),
        np.ones(1000 * sum(gt.size().values())) - 0.1,
        np.zeros(fn_count),
    ])
    
    return average_precision_score(y_true, y_scores)