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

# Intel data
VIDEO_COLLECTION_BASEURL_INTEL = "http://olimar.stanford.edu/hdd/intel_self_driving/" 
VIDEO_METADATA_FILENAME_INTEL = "intel_metadata.json"
req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL_INTEL, VIDEO_METADATA_FILENAME_INTEL), verify=False)
video_collection_intel = sorted(req.json(), key=lambda vm: vm['filename'])

maskrcnn_bbox_files_intel = [ 'maskrcnn_bboxes_0001.pkl', 'maskrcnn_bboxes_0004.pkl' ]

# Names of the cyclist files
cyclist_bbox_files_intel = [ 'cyclist_labels_0001.pkl', 'cyclist_labels_0004.pkl' ]

maskrcnn_bboxes_intel = []
for bbox_file in maskrcnn_bbox_files_intel:
    req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL_INTEL, bbox_file), verify=False)
    maskrcnn_bboxes_intel.append(pickle.loads(req.content))
    
cyclist_bboxes_intel = []
for bbox_file in cyclist_bbox_files_intel:
    req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL_INTEL, bbox_file), verify=False)
    cyclist_bboxes_intel.append(pickle.loads(req.content))
    
video_metadata_intel = [
    VideoMetadata(v["filename"], i, v["fps"], v["num_frames"], v["width"], v["height"])
    for i, v in enumerate(video_collection_intel) if i in [0, 3]
]

def get_bboxes_cydet():
    maskrcnn_bboxes_ism = IntervalSetMapping({
        vm.id: IntervalSet([
            Interval(
                Bounds3D(
                    t1 = frame_num / vm.fps,
                    t2 = (frame_num + 1) / vm.fps,
                    x1 = bbox[0] / vm.width,
                    x2 = bbox[2] / vm.width,
                    y1 = bbox[1] / vm.height,
                    y2 = bbox[3] / vm.height
                ),
                payload = {
                    'class': bbox[4],
                    'score': bbox[5]
                }
            )
            for frame_num, bboxes_in_frame in enumerate(maskrcnn_frame_list)
            for bbox in bboxes_in_frame
        ])
        for vm, maskrcnn_frame_list in zip(video_metadata_intel, maskrcnn_bboxes_intel)
    })
    
    return maskrcnn_bboxes_ism

def get_cyclist_bboxes():
    cyclist_bboxes_ism = IntervalSetMapping({
        vm.id: IntervalSet([
            Interval(
                Bounds3D(
                    t1 = frame_num / vm.fps,
                    t2 = (frame_num + 1) / vm.fps,
                    x1 = bbox[0] / vm.width,
                    x2 = bbox[2] / vm.width,
                    y1 = bbox[1] / vm.height,
                    y2 = bbox[3] / vm.height
                ),
                payload = {
                    'class': bbox[4],
                    'score': bbox[5]
                }
            )
            for frame_num, bboxes_in_frame in enumerate(cyclist_frame_list)
            for bbox in bboxes_in_frame
        ])
        for vm, cyclist_frame_list in zip(video_metadata_intel, cyclist_bboxes_intel)
    })
    
    return cyclist_bboxes_ism

def visualize_cydet(box_list):
    vgrid_spec = VGridSpec(
        video_meta = video_metadata_intel,
        vis_format = VideoBlockFormat(imaps = [
            (str(i), box)
            for i, box in enumerate(box_list)
        ]),
        video_endpoint = VIDEO_COLLECTION_BASEURL_INTEL
    )
    return VGridWidget(vgrid_spec = vgrid_spec.to_json_compressed())

# Parking lot data
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
    
    predictions = predictions.split(
        lambda interval: IntervalSet(
            [interval] if interval['t2'] - interval['t1'] == 30
            else [
                Interval(Bounds3D(
                    t1 = t,
                    t2 = t + 30,
                    x1 = interval['x1'],
                    x2 = interval['x2'],
                    y1 = interval['y1'],
                    y2 = interval['y2']
                ))
                for t in range(int(interval['t1']), int(interval['t2']), 30)
            ]
        )
    )
    
    true_positives = gt.filter_against(
        predictions,
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

def evaluate_on_test(detect_empty_parking_spaces):
    test_gt = get_gt('test')
    test_bboxes = get_bboxes('test')
    reference_video = 1
    
    eps = detect_empty_parking_spaces(test_bboxes, reference_video)
    
    ap = compute_ap(eps, test_gt)
    
    print('Average precision: ', ap)