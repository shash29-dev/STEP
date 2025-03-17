
import json
import os
import pdb
import numpy as np
import pandas as pd
from src.eval.vis_awazi import vis_json_file

class PredTrack(object):
    def __init__(self) -> None:
        self.obji =  {}
        self.preds = {}
        self.gtdata= None
        self.save_dir = None
        self.savefname = None
        
    def add_predictions(self,bbox,kpts,framenum,trck_id,only_frame=False):
        self.preds.setdefault(framenum,{'bbox':[],'kpts':[],'track_id':[],'bbox_bkp':[],'kpts_bkp':[]})
        if not only_frame:
            self.preds[framenum]['bbox'].append(bbox)
            self.preds[framenum]['kpts'].append(kpts)
            self.preds[framenum]['bbox_bkp'].append(bbox)
            self.preds[framenum]['kpts_bkp'].append(kpts)
            self.preds[framenum]['track_id'].append(trck_id)
    
    def save_jsons(self):
        all_data = []
        for framenum,val in self.preds.items():
            file_name = self.gtdata[str(framenum)]['file_name']
            bboxes = val['bbox']
            if len(bboxes)==0:
                all_data.append({'framenum': framenum,
                            'image_file':file_name,
                            'bbox': None,
                            'kpts': None,
                            'track_id': None,
                            })
            else:
                for objid,box in enumerate(bboxes):
                    kpts = self.preds[framenum]['kpts'][objid]
                    track_id = self.preds[framenum]['track_id'][objid]
                    all_data.append({'framenum': framenum,
                                    'image_file':file_name,
                                    'bbox': box,
                                    'kpts': kpts,
                                    'track_id': track_id,
                                    })
            
        savefname = os.path.basename(self.savefname)
        with open(self.save_dir+os.sep+savefname, 'w') as f: json.dump(all_data,f)
        vis_json_file(self.save_dir+os.sep+savefname)
    
     
    def is_frame_tracked(self,bbgt,bbpd,thresh=0.50):
        all_iou= []
        for bbgte in bbgt:
            for bbpde in bbpd:
                all_iou.append(self.calculate_iou(bbgte,bbpde))
                
        detected = (np.array(all_iou)>thresh).sum() > 0
        bpud=None
        if detected:
            idx = all_iou.index(max(all_iou))
            bbpd[idx%len(bbpd)] = bbgt[idx//len(bbpd)]
        
        return detected, all_iou, bbpd
    
    def calculate_iou(self,box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou
    
class EvalUtils(object):
    def __init__(self) -> None:
        pass 
    
    @staticmethod
    def add_offset_to_bbox(bbox,offset_percentage):
        current_center_x = bbox[0] + bbox[2] / 2
        current_center_y = bbox[1] + bbox[3] / 2
        
        min_dimension = min(bbox[2], bbox[3])
        expand_value = min_dimension * offset_percentage 


        expanded_width = bbox[2] + 2 * expand_value
        expanded_height = bbox[3] + 2 * expand_value
        new_bbox = [
                current_center_x - expanded_width / 2,  # xmin
                current_center_y - expanded_height / 2,  # ymin
                expanded_width,  # new width
                expanded_height  # new height
            ]
        
        return new_bbox
