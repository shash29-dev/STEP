import glob
import importlib
import os
import time
import pandas as pd

import imageio
import numpy as np
import matplotlib.pyplot as plt
import pdb
import json
import cv2
import sys
sys.path.append('/media/cvig/A/shashikant/PoseEstimation/AnimalPoseGit/Refactored/STEP/')
# import mmpose.datasets.baseinfo.ap10k_info as mpinfo
# import mmpose.datasets.baseinfo.crowdpose_info as mpinfo
import mmpose.datasets.baseinfo.mousepose_info as mpinfo
# import mmpose.datasets.baseinfo.fishpose_info as mpinfo
from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.datasets.dataset_info import DatasetInfo


def draw_circles_on_image(image, points, radius=10, color=(0, 0, 255), thickness=2):
    image_with_circles = image.copy()

    for point in points:
        center = tuple(point)
        image_with_circles = cv2.circle(image_with_circles, center, radius, color, thickness)

    return image_with_circles

def get_bbox_indicator_vector(points, bbox,offset_factor=0.2):
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    offset_x = offset_factor * bbox_width
    offset_y = offset_factor * bbox_height
    enlarged_bbox = (bbox_x - offset_x, bbox_y - offset_y, bbox_width + 2 * offset_x, bbox_height + 2 * offset_y)
    indicator_vector = np.array([1 if enlarged_bbox[0] <= x <= enlarged_bbox[0] + enlarged_bbox[2] and enlarged_bbox[1] <= y <= enlarged_bbox[1] + enlarged_bbox[3] else 0 for x, y in points])

    # indicator_vector = np.array([1 if bbox_x <= x <= bbox_x + bbox_width and bbox_y <= y <= bbox_y + bbox_height else 0 for x, y in points])
    return indicator_vector

def draw_bounding_boxes(image_file, boxes,kpts,colors,cm,fnum,track_ids_all,dinfo):
    image = cv2.imread(image_file)
    # image=image[:,:,::-1]
    image_with_boxes = np.copy(image)
    track_info = {}
    for boxtid,kpt in zip(boxes,kpts):
        track_id = boxtid[0]
        box = boxtid[1]
        if box is not None:
            x1, y1, w,h = box
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            x2, y2 = x1 + w, y1 + h
            color = colors[track_ids_all.index(track_id)]
            cm_traj = cm[track_id]
            find_till_frame = (cm_traj['framenum'] == fnum).idxmax()
            track_info[track_id] = cm_traj.loc[:find_till_frame]
            kpt =np.array(kpt).reshape(-1,3)[None].astype(np.float32)
            pinside = get_bbox_indicator_vector(kpt[0,:,:2],box)
            kpt[:,:,-1] = kpt[:,:,-1] * pinside
            
            # pdb.set_trace()
            # cv2.imwrite('tmp.png',image_with_boxes)

            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            image_with_boxes = imshow_keypoints(image_with_boxes,kpt,kpt_score_thr=0.1,skeleton=dinfo.skeleton,pose_kpt_color=dinfo.pose_kpt_color,pose_link_color=dinfo.pose_link_color, radius=5,thickness=2)
    return image_with_boxes, track_info

def draw_trajectory_on_image(image, df,color):
    for i in range(1, len(df)):
        start_point = (int(df['center_x'].iloc[i - 1]), int(df['center_y'].iloc[i - 1]))
        end_point = (int(df['center_x'].iloc[i]), int(df['center_y'].iloc[i]))
        color = color
        thickness = 2
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image

def draw_full_trajectory_on_image(image, aif,colors,track_ids_all):
    for key in aif.keys():
        color=  colors[track_ids_all.index(key)]
        df = pd.DataFrame(aif[key])
        for i in range(1, len(df)):
            start_point = (int(df['center_x'].iloc[i - 1]), int(df['center_y'].iloc[i - 1]))
            end_point = (int(df['center_x'].iloc[i]), int(df['center_y'].iloc[i]))
            color = color
            thickness = 2
            image = cv2.line(image, start_point, end_point, color, thickness)
    return image

def get_center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return pd.Series([center_x, center_y], index=['center_x', 'center_y'])


def vis_json_file(filename,awazi_fol = '/media/cvig/A/shashikant/PoseEstimation/AnimalPoseGit/Evaluation/awazi/'):
    with open(filename, 'r') as f: data = json.load(f)
    
    # dataset_info_path = 'mmpose/datasets/baseinfo/ap10k_info.py'
    # dataset_info_path = 'mmpose/datasets/baseinfo/crowdpose_info.py'
    dataset_info_path = 'mmpose/datasets/baseinfo/mousepose_info.py'
    # dataset_info_path = 'mmpose/datasets/baseinfo/fishpose_info.py'
    dcname = dataset_info_path.split(os.sep)[-1].split('.')[0]
    # pdb.set_trace()
    # spec = importlib.util.spec_from_file_location(dcname, dataset_info_path)
    # module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module)
    dataset_info = getattr(mpinfo,dcname)
    dataset_info = DatasetInfo(dataset_info)
    
    colors_lst = [
     (0, 0, 255),(255, 0, 0),(0, 255, 0),
     (0, 255, 255),(255, 0, 255),(255, 255, 0)
    ]
    colors = []
   
    df = pd.DataFrame(data)
    track_ids_all = pd.unique(df['track_id']).tolist()
    frames = df.groupby('framenum')
    for idx in range(len(track_ids_all)):
        colors.append(colors_lst[idx%len(colors_lst)])
    
    track_group = df.groupby('track_id')
    centers_movement = {}
    all_track_info = {}
    for tid, group in track_group:
        centers = group.bbox.apply(get_center)
        centers['image_file'] = group['image_file']
        centers['framenum'] = group['framenum']
        centers_movement[tid] = centers
        # image = group['image_file'][0]
        # image= cv2.imread(awazi_fol + image)
        # x=draw_trajectory_on_image(image,centers,color=colors[tid])
        # pdb.set_trace()
        
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # out = cv2.VideoWriter('tmp.mp4',fourcc,20,(1920,1080))
    writer = imageio.get_writer(filename.replace('.json','.mp4'))
    ims_mp4 = [] 
    idxcnt = 0
    for fnum,group in frames:
        idxcnt += 1
        # kk=len(frames)-500
        kk=0
        if idxcnt<kk: continue
        if idxcnt>kk+100: break
        print('Writing... fnum : {}|{}|{}'.format(fnum,idxcnt,len(df)))
        bboxes=[]
        kpts=[]
        for idx,row in group.iterrows():
            bboxes.append([row.track_id,row.bbox])
            kpts.append(row.kpts)
            
        # image_file = awazi_fol + row.image_file
        image_file = row.image_file
        image_file = image_file.replace('/home/cvig/shashikant/', '/media/shashikant/A/shashikant/')
        imbbox,tinfo = draw_bounding_boxes(image_file,bboxes,kpts,colors,centers_movement,fnum,track_ids_all,dataset_info)
        imbbox_track = imbbox
        imbbox_track = imbbox[:,:,::-1]
        all_track_info.update(tinfo)
        # cv2.imwrite('tmp1.png',imbbox_track[:,:,::-1])
        # imbbox_track = draw_full_trajectory_on_image(imbbox,all_track_info,colors,track_ids_all)
        # out.write(imbbox_track)
        writer.append_data(imbbox_track)
        # cv2.imshow('Frame',cv2.resize(imbbox_track,(1920//2,1080//2))[:,:,::-1])
        # if cv2.waitKey(1) == ord('1'):
        #     break
    #     ims_mp4.append(imbbox_track)
    # imageio.mimsave('tmp.mp4',ims_mp4)
    # out.release()
    # imbbox_track = draw_full_trajectory_on_image(imbbox,all_track_info,colors,track_ids_all)
    writer.append_data(imbbox_track)
    writer.close()
    # cv2.destroyAllWindows()
        
if __name__=='__main__':
    afn= '/media/cvig/A/shashikant/PoseEstimation/AnimalPoseGit/Refactored/STEP/logs/ref_step/snaps/images/crowdpose_nokpts_kptsemb/awazi/vid_50.json'
    awazi_fol = '/media/cvig/A/shashikant/PoseEstimation/AnimalPoseGit/Evaluation/awazi/'
    for filename in glob.glob(afn):
        vis_json_file(filename,awazi_fol=awazi_fol)