import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy/rekall/bounds')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy/rekall')


from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat
from rekall.predicates import *
from rekall.bounds import Bounds, utils


DEV_SET = '0002.mp4'
IOU_THRESHOLD = 0.5

def sort_score(intrvl):
    return intrvl['payload']['score']

def calculate_ap(constructed_labels, gt):
    #intervals_list = constructed_labels[DEV_SET].get_intervals()
    intervals_list = list(set(constructed_labels[DEV_SET].get_intervals()))
    #sort by score
    intervals_list.sort(key=sort_score, reverse=True)
    print('generated labels: {}'.format(len(intervals_list))) 
    #print(intervals_list[:100])
    
    #construct precision + recall vectors
    gt_total_positives = gt[DEV_SET].size()
    print('gt labels: {}'.format(gt_total_positives))
    
    #iterate through join test and calc precision + recall
    tp_count_prec = 0
    tp_count_rec = 0
    index = 0
    precision = []
    recall = []
    gt_bboxes = []
    cons_bounds = []
    
    '''for index, bbox in enumerate(intervals_list, start=1):
        gt_box_matches = bbox['payload']['gt_interval']
        if (bbox['payload']['true_positive']):
            tp_count_prec += 1
            print (len(gt_box_matches))
            for gt_box in gt_box_matches:
                if gt_box not in gt_bboxes:
                    tp_count_rec += 1
                    if bbox['payload']['gt_interval'] is not None:
                        gt_bboxes.append(gt_box)'''

    for bbox in intervals_list:
        already_exists = False
        if bbox['bounds'] not in cons_bounds:
            index += 1 #only increment index if it not a duplicate tp
            cons_bounds.append(bbox['bounds'])
        else:
            already_exists = True
        if (bbox['payload']['true_positive']):
            if already_exists is False:
                tp_count_prec += 1
            gt_box = bbox['payload']['gt_interval']
            if gt_box not in gt_bboxes:
                tp_count_rec += 1
                if bbox['payload']['gt_interval'] is not None:
                    gt_bboxes.append(gt_box)

        prec = tp_count_prec/index
        rec = tp_count_rec/gt_total_positives
        precision.append(prec)
        recall.append(rec)
          
    plt.plot(recall, precision) 
    print (rec)
    #auc = metrics.auc(recall, precision)
    auc = CalculateAveragePrecision(recall, precision)[0]
    #auc = ElevenPointInterpolatedAP(recall, precision)[0]
    print('average precision @ IOU of {}:  {}'.format(IOU_THRESHOLD, auc))
    return (auc, IntervalSet(gt_bboxes))

#https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]