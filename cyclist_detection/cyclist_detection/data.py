import sys
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/vgrid/vgridpy')


import urllib3, requests, json, os, pickle
from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat
from rekall.predicates import *


#Obviates the appearance of a certificate warning

urllib3.disable_warnings()

# Names of the maskrcnn files
maskrcnn_bbox_files = [ 'maskrcnn_bboxes_0001.pkl', 'maskrcnn_bboxes_0002.pkl', 'maskrcnn_bboxes_0003.pkl',
                'maskrcnn_bboxes_0004.pkl', 'maskrcnn_bboxes_0005.pkl' ]

# Names of the cyclist files
cyclist_bbox_files = [ 'cyclist_labels_0001.pkl', 'cyclist_labels_0002.pkl', 'cyclist_labels_0003.pkl',
                'cyclist_labels_0004.pkl', 'cyclist_labels_0005.pkl' ]

# location of the video metadata file.
# It is assumed that video data is located relative to this file.
VIDEO_COLLECTION_BASEURL = "http://olimar.stanford.edu/hdd/intel_self_driving/" 
VIDEO_METADATA_FILENAME = "intel_metadata.json"

def load_data():
    # Grab the metadata (width, height, number of frames, FPS) of my video collection from olimar
    req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, VIDEO_METADATA_FILENAME), verify=False)
    video_collection = sorted(req.json(), key=lambda vm: vm['filename'])

    maskrcnn_bboxes = []
    for bbox_file in maskrcnn_bbox_files:
        req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, bbox_file), verify=False)
        maskrcnn_bboxes.append(pickle.loads(req.content))
    
    cyclist_bboxes = []
    for bbox_file in cyclist_bbox_files:
        req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, bbox_file), verify=False)
        cyclist_bboxes.append(pickle.loads(req.content))

    # Load the video metadata into VideoMetadata objects, using filename for the id
    video_metadata = [
        VideoMetadata(v["filename"], v["filename"], v["fps"], v["num_frames"], v["width"], v["height"])
        for v in video_collection
    ]

    # Load the maskrcnn bboxes into Rekall, using video id as key
    # Units of Bounds are seconds for time, relative units for X and Y
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
                    'score': bbox[5],
                    'iou': 0
                }
            )
            for frame_num, bboxes_in_frame in enumerate(maskrcnn_frame_list)
            for bbox in bboxes_in_frame
        ])
        for vm, maskrcnn_frame_list in zip(video_metadata, maskrcnn_bboxes)
    })

    # Load the cyclist bboxes into Rekall, using video id as key
    # Units of Bounds are seconds for time, relative units for X and Y
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
                    'score': bbox[5],
                    'iou' : 0
                }
            )
            for frame_num, bboxes_in_frame in enumerate(cyclist_frame_list)
            for bbox in bboxes_in_frame
        ])
        for vm, cyclist_frame_list in zip(video_metadata, cyclist_bboxes)
    })
    return [maskrcnn_bboxes_ism, cyclist_bboxes_ism]

def get_people_and_bicycle_ism(maskrcnn_bboxes_ism):
    object_names = ["person", "bicycle"]
    object_isms = [
        maskrcnn_bboxes_ism.filter(lambda interval: interval['payload']['class'] == object_name)
        for object_name in object_names
    ]
    return [object_isms[0], object_isms[1]]




