import sys
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy')
sys.path.append('/Users/avanikanarayan/Documents/Stanford/research/hazy_research/rekall_dev/rekall/rekallpy/rekall/bounds')


from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from rekall.bounds import Bounds, utils


def track_cyclist_overtime(ism, REMOVE_BLIPS):
    
    #COALESCE THE CONSTRUCTED CYCLIST BBOXES TO TRACK A BIKER OVER TIME
    def IOU_track(bbox1, bbox2):
        IOU_THRESHOLD = .50
        area1 = area(bbox1)
        area2 = area(bbox2)
        
        intersection_box = bbox1.combine_per_axis(bbox2, utils.bounds_intersect, utils.bounds_intersect, utils.bounds_intersect) 
        overlap_area = _width(intersection_box) * _height(intersection_box)
        iou = overlap_area / (area1 + area2 - overlap_area)
    
        if iou >= IOU_THRESHOLD:
            return True
        else:
            return False
        
    def touching():
        return lambda intrvl1, intrvl2: intrvl1['t2'] == intrvl2['t1'] 
    
    def area_same(box1, box2):
        epsilon = .3
        area1 = area(bbox1)
        area2 = area(bbox2)
        if abs(area1 - area2) <= epsilon:
            return True
        else: 
            return False

    def join_over_time_predicate(intrvl1, intrvl2):
        if Bounds3D.T(touching())(intrvl1,intrvl2) or IOU_track(intrvl1['bounds'], intrvl2['bounds']):
            return True
        else:
            return False 
        
    def bounds_merge(intrvl1, intrvl2):
        bounds = (intrvl1['t1'], intrvl2['t2'], intrvl2['x1'], intrvl2['x2'], intrvl2['y1'], intrvl2['y2'] )
        return Bounds3D.fromTuple(bounds)

    def payload_merge(payload1, payload2):
        return payload1.union(payload2)

    #first need to map interval its payload
    mapped_ism = ism.map(lambda intrvl: Interval(intrvl['bounds'], IntervalSet([intrvl])))
    coalesce_cyclist_over_time = mapped_ism.coalesce(('t1', 't2'), bounds_merge, payload_merge, predicate=join_over_time_predicate, epsilon = 5)
    if REMOVE_BLIPS:
        final_coalesce_cyclist_over_time = remove_blips(coalesce_cyclist_over_time)
    else:
        final_coalesce_cyclist_over_time = coalesce_cyclist_over_time
    return final_coalesce_cyclist_over_time


def remove_blips(ism):
    temp_ism = ism
    new_coalesced = dict()
    for k, intervalset in temp_ism.get_grouped_intervals().items():
        new_interval_set = []
        for intrvl in intervalset.get_intervals():
            new_intrvl = Interval(intrvl['bounds'], intrvl['payload'])
            first_interval = new_intrvl['payload'].get_intervals()[0]
            if len(new_intrvl['payload'].get_intervals()) < 3 and first_interval['payload']['class'] == 'person':
                continue
            else:
                new_interval_set.append(new_intrvl)
        new_coalesced[k] = IntervalSet(new_interval_set)

    new_coalesced = IntervalSetMapping(new_coalesced)
    return new_coalesced  

def remove_flickering(ism, video):
    # FIX FLICKERING -- INSERT BOUNDING BOXES IN INTERVALS WHERE FLICKERING IS PRESENT
    def sort_t1(intrvl):
        return intrvl['t1']

    #code to insert bounding boxes when flickering occurs:
    grouped_intervals = ism[video].get_intervals()
    new_added_intervals = []
    for g_volume in grouped_intervals:
        all_nested_intervals = g_volume
        if len(g_volume['payload']) > 1:
            nested_intervals = g_volume['payload'].get_intervals()
            nested_intervals.sort(key=sort_t1)
            for index, nested_intrvl in enumerate(nested_intervals):
                if index != 0:
                    if nested_intrvl['t1'] != prev['t1'] and nested_intrvl['t2'] != prev['t2'] and nested_intrvl['t1'] != prev['t2']:
                        t1 = round(prev['t2'], 1)
                        t2 = round(t1 + .1, 1)
                        x1 = (0.5)*(prev['x1'] + nested_intrvl['x1'])
                        x2 = (0.5)*(prev['x2'] + nested_intrvl['x2'])
                        y1 = (0.5)*(prev['y1'] + nested_intrvl['y1'])
                        y2 = (0.5)*(prev['y2'] + nested_intrvl['y2'])
                        while t2 <= nested_intrvl['t1']:
                            new_bounds = Bounds3D.fromTuple((t1,t2,x1,x2,y1,y2))
                            nested_intrvl['payload']['class'] = "ghost_box"
                            new_interval = IntervalSet([Interval(new_bounds, nested_intrvl['payload'])])
                            all_nested_intervals['payload'] = all_nested_intervals['payload'].union(new_interval)
                            t1 += .1
                            t2 += .1
                            t1 = round(t1, 1)
                            t2 = round(t2, 1)
                        
                prev = nested_intrvl         
        new_added_intervals.append(all_nested_intervals)

    return new_added_intervals
    
#HELPER FUNCTIONS
def area(bbox):
    return (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])

def _width(bbox):
    return bbox['x2'] - bbox['x1']

def _height(bbox):
    return bbox['y2'] - bbox['y1']
