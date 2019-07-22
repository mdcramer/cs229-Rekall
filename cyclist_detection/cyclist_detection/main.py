import sys
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy/rekall/bounds')


from data import *
from bboxes import *
from flickering import *
from metrics import *

import numpy as np
import matplotlib.pyplot as plt
from rekall.bounds import Bounds, utils
from rekall.predicates import *

PERSON_BBOX = True
BICYCLE_BBOX = False
DEV_SET = '0002.mp4'
TEST_SET = '0001.mp4'
IOU_THRESHOLD = 0.5
FLICKERING = True


#CONSTRUCTES THE GT CYCLIST DEV BBOXES AND CONSTRUCTED CYCLIST BBOXES
def reset_dev_and_test_set(cyclist_bboxes_ism, unwrapped_constructed_bboxes):

    cyclist_bboxes_dev = cyclist_bboxes_ism.filter(
            lambda intrvl: intrvl['payload']['class'] == 'Cyclist')
  
    constructed_cyclist_bboxes_dev = unwrapped_constructed_bboxes

    #reset all iou values to 0
    def reset_iou(interval_set):
        for nested_intrvl in interval_set:
            nested_intrvl['payload']['iou'] = 0
            nested_intrvl['payload']['true_positive'] = False
            nested_intrvl['payload']['gt_interval'] = None
        return interval_set

    cyclist_bboxes_dev[DEV_SET] = IntervalSet(reset_iou(cyclist_bboxes_dev[DEV_SET].get_intervals()))
    constructed_cyclist_bboxes_dev[DEV_SET] = IntervalSet(reset_iou(constructed_cyclist_bboxes_dev[DEV_SET].get_intervals()))

    dev_constructed = IntervalSetMapping({
        DEV_SET : constructed_cyclist_bboxes_dev[DEV_SET]
    })

    dev_ground_truth = IntervalSetMapping({
        DEV_SET : cyclist_bboxes_dev[DEV_SET]
    })
    
    return [dev_constructed, dev_ground_truth]

def get_true_positives(dev_constructed, dev_ground_truth, IOU_THRESHOLD):
    
    def get_iou(intrvl, gt_box):
        gt_area = area(gt_box)
        test_area = area(intrvl)
        test_box_bounds = intrvl['bounds']
        gt_box_bounds = gt_box['bounds']

        intersection_box = test_box_bounds.combine_per_axis(gt_box_bounds, utils.bounds_intersect, utils.bounds_intersect, utils.bounds_intersect) 
        overlap_area = _width(intersection_box) * _height(intersection_box)
        iou = overlap_area / (gt_area + test_area - overlap_area)
        
        return iou


    def bbox_overlap(intrvl1, intrvl2):
        iou = get_iou(intrvl1, intrvl2)
        if iou > IOU_THRESHOLD:
            return True
        else:
            return False

    def modified_interval(intrvl1, intrvl2):
        new_intrvl = Interval(intrvl1['bounds'], intrvl1['payload'])
        new_intrvl['payload']['true_positive'] = True
        new_intrvl['payload']['gt_interval'] = intrvl2
        return new_intrvl

    tp = dev_constructed.join(
        dev_ground_truth,
        predicate = and_pred(
            Bounds3D.T(equal()), # equal/overlaps along the time dimension
            Bounds3D.X(overlaps()), # boxes overlap in the X dimension
            Bounds3D.Y(overlaps()),
            bbox_overlap
            
        ), 
        merge_op = modified_interval,
        window=0.0 #CHECK THIS
    )

    no_dups = IntervalSet((list(set(tp[DEV_SET].get_intervals()))))
    no_dups_ism = IntervalSetMapping({DEV_SET: no_dups})

    return no_dups_ism

def construct_entire_labeled_set(true_positives, dev_constructed):
    true_positive_intervals = true_positives[DEV_SET].get_intervals()
    all_dev_constructed = dev_constructed[DEV_SET].get_intervals()

    all_tp_bounds = [x['bounds'] for x in true_positive_intervals]

    false_positives_intervals = []
    for intrvl in all_dev_constructed:
        if intrvl['bounds'] not in all_tp_bounds:
            false_positives_intervals.append(intrvl)
    
    total_generated_labels = list(set(true_positive_intervals + false_positives_intervals))
    return IntervalSetMapping({DEV_SET: IntervalSet(total_generated_labels)})


#HELPER FUNCTIONS
def area(bbox):
    return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])

def _width(bbox):
    return bbox['x2'] - bbox['x1']

def _height(bbox):
    return bbox['y2'] - bbox['y1']


if __name__ == "__main__":
    print ("PERSON_BBOX : {}, BICYCLE_BBOX : {}, FLICKERING: {}".format(PERSON_BBOX, BICYCLE_BBOX, FLICKERING))
    [maskrcnn_bboxes_ism, cyclist_bboxes_ism] = load_data()
    [person_ism, bicycle_ism] = get_people_and_bicycle_ism(maskrcnn_bboxes_ism)
    constructed_cyclist_bboxes = construct_cyclist_bboxes(person_ism, bicycle_ism)

    if PERSON_BBOX:
        constructed_person_bboxes = construct_person_bboxes(constructed_cyclist_bboxes, person_ism, bicycle_ism)
    if BICYCLE_BBOX:
        constructed_bicycle_bboxes = construct_bicycle_bboxes(bicycle_ism)

    #join ism sets to get complete construced set
    if PERSON_BBOX and BICYCLE_BBOX:
        constructed_cyclist_p_boxes = constructed_cyclist_bboxes.union(constructed_person_bboxes)
        constructed_cyclist_total_boxes = constructed_cyclist_p_boxes.union(constructed_bicycle_bboxes)
    elif PERSON_BBOX and not BICYCLE_BBOX:
        constructed_cyclist_total_boxes = constructed_cyclist_bboxes.union(constructed_person_bboxes)
    elif BICYCLE_BBOX and not PERSON_BBOX:
        constructed_cyclist_total_boxes = constructed_cyclist_bboxes.union(constructed_bicycle_bboxes)
    else:
        constructed_cyclist_total_boxes = constructed_cyclist_bboxes

    #coalesce to eliminate duplicates in the ground truth cyclist bboxes and the constructed bboxes
    final_constructed_cyclist_total_bboxes = remove_dup_bboxes(constructed_cyclist_total_boxes, .50)
    final_ground_truth_bboxes = remove_dup_bboxes(cyclist_bboxes_ism, .70)

    #flickering
    coalesced_volumes = track_cyclist_overtime(final_constructed_cyclist_total_bboxes, False)
    coalesced_dev_set_volume = remove_flickering(coalesced_volumes, DEV_SET)

    #unwrap the coalesced_dev_set_volume to get all constructed bboxes
    unwrapped_constructed_bboxes = []
    for intrvl in coalesced_dev_set_volume:
        for nested_interval in intrvl['payload'].get_intervals():
            unwrapped_constructed_bboxes.append(nested_interval)

    unwrapped_constructed_bboxes_ism = IntervalSetMapping({DEV_SET : IntervalSet(unwrapped_constructed_bboxes)})

    #construct ism for dev set - w/flickering correction
    if FLICKERING:
        [dev_constructed, dev_ground_truth] = reset_dev_and_test_set(final_ground_truth_bboxes, unwrapped_constructed_bboxes_ism)
    else:
        [dev_constructed, dev_ground_truth] = reset_dev_and_test_set(final_ground_truth_bboxes, final_constructed_cyclist_total_bboxes)
    #get true positives (no duplicates)
    true_positives = get_true_positives(dev_constructed, dev_ground_truth, IOU_THRESHOLD)

    #get entire labeled set
    final_labeled = construct_entire_labeled_set(true_positives, dev_constructed)
    '''print(dev_constructed.size())
    print(final_labeled.size())'''

    #calculate ap
    [auc, gg_bboxes_recognized] = calculate_ap(final_labeled, dev_ground_truth)

    







    







