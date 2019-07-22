import sys
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall/rekall/rekallpy')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall/rekall/rekallpy/rekall/bounds')

import copy

from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat
from rekall.predicates import *
from rekall.bounds import Bounds, utils

DEV_SET = '0002.mp4'

def construct_cyclist_bboxes(person_ism, bicycle_ism):
    constructed_cyclist_bboxes = person_ism.join(
        bicycle_ism,
        predicate = and_pred(
            Bounds3D.T(equal()), # equal along the time dimension
            Bounds3D.X(overlaps()), # boxes overlap in the X dimension
            Bounds3D.Y(overlaps()) # boxes overlap in the Y dimension
        ),
        merge_op = lambda person, bicycle: Interval(
            person['bounds'].span(bicycle['bounds']), # We use the "span" method of Bounds3D to get a spanning bound
            payload = {
                'class': 'bike_person_cyclist',
                'score' : (0.5)*(person['payload']['score'] + bicycle['payload']['score']),
                'iou' : 0,
            }
        ),
        window = 0.5 # choose only pairs that differ by less than half a second from each other
    )
    return constructed_cyclist_bboxes

def construct_person_bboxes(constructed_cyclist_bboxes, person_ism, bicycle_ism, modify_size = False):
    #ADD CASE WITH PERSON (THAT FULLFILLS HEURISTICS) BUT NO BICYCLE
    def filter_size_location(intrvl):
        intervals = constructed_cyclist_bboxes[DEV_SET].get_intervals()
        for interval in intervals:
            if interval['t1'] == intrvl['t1'] and interval['t2'] == intrvl['t2']:
                return False
        width = intrvl['x2'] - intrvl['x1']
        height = intrvl['y2'] - intrvl['y1']
        x1 = intrvl['x1']
        y1 = intrvl['y1']
        if width <= .05 and height <= .1 and x1 <= .45 and x1 >= .2 and y1 >= .2 and y1 <= .6:
            return True
        else:
            return False
    
    #TODO: Maybe we don't want this??
    #person_wo_bicycle = person_ism.minus(bicycle_ism)
    constructed_person_bboxes = person_ism.filter(filter_size_location)
    #modifies shape of bboxes to match more gt bboxes
    modify_size = True
    if modify_size:
        modified_constructed_person_bboxes = modify_person_bboxes(constructed_person_bboxes)
        return modified_constructed_person_bboxes
    return constructed_person_bboxes

def construct_bicycle_bboxes(bicycle_ism):
    temp_bicycle_bboxes= bicycle_ism
    new_bicycle_bboxes = dict()
    for k, intervalset in temp_bicycle_bboxes.get_grouped_intervals().items():
        new_interval_set = []
        for intrvl in intervalset.get_intervals():
            new_intrvl = Interval(intrvl['bounds'], intrvl['payload'])
            new_intrvl['y1'] = new_intrvl['y1'] - ((new_intrvl['y2'] - new_intrvl['y1'])/2)
            #new_intrvl['x1'] -= .1
            #new_intrvl['x2'] += .1
            new_interval_set.append(new_intrvl)
        new_bicycle_bboxes[k] = IntervalSet(new_interval_set)

    new_bicycle_bboxes = IntervalSetMapping(new_bicycle_bboxes)
    return new_bicycle_bboxes    

def modify_person_bboxes(constructed_person_bboxes):
    temp_constructed_cyclist_additional = constructed_person_bboxes
    new_constructed_cyclist_additional = dict()
    for k, intervalset in temp_constructed_cyclist_additional.get_grouped_intervals().items():
        new_interval_set = []
        for intrvl in intervalset.get_intervals():
            new_intrvl = Interval(intrvl['bounds'], intrvl['payload'])
            new_intrvl['x1'] -= .0025
            new_intrvl['x2'] += .0025
            new_intrvl['y2'] -= .01
            new_interval_set.append(new_intrvl)
        new_constructed_cyclist_additional[k] = IntervalSet(new_interval_set)

    new_constructed_cyclist_additional = IntervalSetMapping(new_constructed_cyclist_additional)
    return new_constructed_cyclist_additional


def remove_dup_bboxes(ism, IOU_THRESHOLD):
    #HELPER FUNCTIONS
    def area(bbox):
        return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])

    def _width(bbox):
        return bbox['x2'] - bbox['x1']

    def _height(bbox):
        return bbox['y2'] - bbox['y1']

    def bounds_merge_intervals(intrvl1, intrvl2):
        #assuming that 't1' and 't2' are the same for both
        t1 = intrvl1['t1']
        t2 = intrvl1['t2']
        x1, x2, y1, y2 = [0,0,0,0]
        
        area1 = area(intrvl1)
        area2 = area(intrvl2)
        if area1 >= area2:
            x1, x2, y1, y2 = [intrvl1['x1'], intrvl1['x2'], intrvl1['y1'], intrvl1['y2']]
        else:
            x1, x2, y1, y2 = [intrvl2['x1'], intrvl2['x2'], intrvl2['y1'], intrvl2['y2']]
        return Bounds3D.fromTuple((t1,t2,x1,x2,y1,y2))
        
    def iou_threshold(bbox1, bbox2):
        area1 = area(bbox1)
        area2 = area(bbox2)
        
        if inside()(bbox1, bbox2) or inside()(bbox2, bbox1):
            return True
        
        intersection_box = bbox1.combine_per_axis(bbox2, utils.bounds_intersect, utils.bounds_intersect, utils.bounds_intersect) 
        overlap_area = _width(intersection_box) * _height(intersection_box)
        iou = overlap_area / (area1 + area2 - overlap_area)
    
        if iou >= IOU_THRESHOLD:
            return True
        else:
            return False
        
    def overlapping_bboxes_predicate(intrvl1, intrvl2):
        if Bounds3D.T(equal())(intrvl1,intrvl2) and iou_threshold(intrvl1['bounds'], intrvl2['bounds']):
            return True
        else:
            return False

    coalesced_ism = ism.coalesce(('t1', 't2'), bounds_merge_intervals, predicate=overlapping_bboxes_predicate)
    return coalesced_ism



