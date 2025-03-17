import pdb
import cv2
import numpy as np
import mmcv
import pandas as pd
import matplotlib.pyplot as plt
import random
import colorsys
import seaborn as sns
from mmpose.core import imshow_bboxes, imshow_keypoints
import imageio
import os 

def generate_n_random_dark_colors(n):
    # colors_lst  = ['blue','red','green','cyan','magenta','yellow']
    colors_lst  = ['blue','red','cyan','magenta','yellow','green','white','black',]
    colors = []
    for idx in range(n):
        colors.append(colors_lst[idx%len(colors_lst)])
    return colors

def generate_n_random_dark_colors_rgb(n):
    # colors_lst  = ['blue','red','green','cyan','magenta','yellow']
    colors_lst  = [(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]
    colors = []
    for idx in range(n):
        colors.append(colors_lst[idx%len(colors_lst)])
    return colors

def get_center(bbox):
    x, y, w, h = bbox
    w= w-x
    h = h-y
    center_x = x + w / 2
    center_y = y + h / 2
    return pd.Series([center_x, center_y], index=['center_x', 'center_y'])


def save_gif(df, dinfo,cfg):
    vids = pd.unique(df.video_id)
    for idx_,vid in enumerate(vids):
        print('Writing... {}/{}'.format(idx_,len(vids)))
        video = df[df.video_id==vid]
        group = video.groupby('image_file', group_keys=False)
        group_dict = group.groups
        num_obj = pd.unique(video.track_id)
        color_gt = generate_n_random_dark_colors(len(num_obj))
        # color_pd = generate_n_random_dark_colors(len(num_obj))
        
        all_frames= []
        for imfile, indices in group_dict.items():
            imfile = imfile.replace('data/','')
            bgto = plt.imread(imfile)
            bpdo = plt.imread(imfile)
            for idx in indices:
                visdata = video.loc[idx]
                tid = num_obj.tolist().index(visdata.track_id)
                gtbox = np.array(visdata.orim_bbox_gt)[None]
                pdbox = np.array(visdata.orim_bbox)[None]
                gt_kpts = np.array(visdata.orim_kpts_gt).reshape(-1,3)[None]
                pd_kpts = np.array(visdata.orim_kpts).reshape(-1,3)[None]
                pd_kpts[:,:,-1]=1
                bgto=mmcv.imshow_bboxes(bgto.copy(),gtbox,colors=color_gt[tid],thickness=2,show=False)
                bpdo=mmcv.imshow_bboxes(bpdo.copy(),pdbox,colors=color_gt[tid],thickness=2,show=False)
                bgto = imshow_keypoints(bgto,gt_kpts,skeleton=dinfo.skeleton,pose_kpt_color=dinfo.pose_kpt_color,pose_link_color=dinfo.pose_link_color, radius=5,thickness=5)
                bpdo = imshow_keypoints(bpdo,pd_kpts,skeleton=dinfo.skeleton,pose_kpt_color=dinfo.pose_kpt_color,pose_link_color=dinfo.pose_link_color, radius=5,thickness=5)
               
            kpim= np.hstack((bgto,bpdo))
            # kpim=bpdo
            all_frames.append(kpim) 
        print(len(all_frames))
        # track_group = video.groupby('track_id')
        # centers_movement = {}
        # all_track_info = {}
        # for tid, group in track_group:
        #     centers = group.orim_bbox.apply(get_center)
        #     centers_movement[tid] = centers    
        
        # color_gt= generate_n_random_dark_colors_rgb(len(num_obj))
        # for key in centers_movement.keys():
        #     color = color_gt[key-1]
        #     dff = pd.DataFrame(centers_movement[key])
        #     for i in range(1, len(dff)):
        #         start_point = (int(dff['center_x'].iloc[i - 1]), int(dff['center_y'].iloc[i - 1]))
        #         end_point = (int(dff['center_x'].iloc[i]), int(dff['center_y'].iloc[i]))
        #         colorf = (color[2],color[0],color[1])
        #         thickness = 8
        #         image = cv2.line(all_frames[7], start_point, end_point, colorf, thickness)
        output_dir =  cfg.snaps.image_save_dir
        gifname = 'animal_{}.mp4'.format(vid)
        # gifname = 'animal.mp4'
        gifname = os.path.join(output_dir,gifname)
        os.makedirs(os.path.dirname(gifname),exist_ok=True)
        imageio.mimsave(gifname,all_frames)
        pass
        # imageio.mimsave(gifname,all_frames,duration=1000 * 1/15,loop=0)