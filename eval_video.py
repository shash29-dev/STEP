import numpy as np
import pdb

import torch
from omegaconf import OmegaConf,open_dict
# import src.models.lwl.lwl_net as lwl_networks
# import src.models.lwl.lwl_box_net as lwl_box_networks
# from src.models.loss.segmentation import LovaszSegLoss
import torch.optim as optim
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes
import os
import torchvision
from torchvision.utils import draw_keypoints
import time
try:
    from src.models.tracking import tompnet
    import src.utils.utilfunc as mut
    import src.actors.tracking as actors
    import src.eval.evalutils_v1 as eut_v1
except:
    pass
from src.models.loss.bbr_loss import GIoULoss, KpMSELoss
import src.models.loss as ltr_losses
from src.trainers.ltr_trainer import LTRTrainer
import json
import pandas as pd
import src.eval.evalutils as eut

from data.tensordict import TensorDict
from data.mmpipeline_trans import train_pipeline, val_pipeline, test_pipeline
from mmpose.datasets.dataset_info import DatasetInfo
import importlib
import cv2

def evaluation_loop(cfg):
    cfd = cfg.data
    settings= cfg.data.settings
    device= 'cuda'

    nkp =10
    video_path = cfg.pipeline.video_path
    afd = eut.get_frames_from_video(video_path,cfg,pkpts=False,numframes=None)
    num_train_frames= 2

    afd= mut.EasyDict(afd)
        
    train_data = {}
    test_data = {}
    for key,value in afd.items():
        train_data[key] = [value[0],value[0]]
        test_data[key] = value  # data_processing don't work if test_images>1. Use test_kpts for visualization OR Correct test_kpts_target, test_kpts_label.. Find issue.
        # test_data[key] = value[num_train_frames:num_train_frames+1]  # For Ex. Check this. It works with test_kpts_target, test_kpts_label.

    train_data = mut.EasyDict(train_data)
    test_data = mut.EasyDict(test_data)
    data = TensorDict (
        {
            'train_images': train_data.all_images,
            'train_bbox': train_data.all_bbox,
            'test_images': test_data.all_images,
        }
    )
    if 'all_kpts' in train_data.keys():
        data['train_kpts'] = train_data.all_kpts

    with open_dict(settings):
        settings.output_sz = settings.feature_sz * 16     # Size of input image crop
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 
                    'sigma_factor': output_sigma, 
                    'kernel_sz': settings.target_filter_sz}
    
    test_pipeline_= test_pipeline
    val_pipeline_ = val_pipeline
    ann_info ={}
    ann_info['image_size'] = np.array([cfd.settings.output_sz,cfd.settings.output_sz])
    ann_info['heatmap_size'] = ann_info['image_size']
    ann_info['use_different_joint_weights'] = cfd.settings.get('use_different_joint_weights', False)
    ann_info['num_joints'] = cfd.settings.num_kpts
    ann_info['num_output_channels'] = cfd.settings.num_kpts
    dataset_info_path = cfd.settings.dataset_info
    dcname = dataset_info_path.split(os.sep)[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(dcname, dataset_info_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_info = getattr(module,dcname)

    dataset_info = DatasetInfo(dataset_info)
    assert ann_info['num_joints'] == dataset_info.keypoint_num
    ann_info['flip_pairs'] = dataset_info.flip_pairs
    ann_info['flip_index'] = dataset_info.flip_index
    ann_info['upper_body_ids'] = dataset_info.upper_body_ids
    ann_info['lower_body_ids'] = dataset_info.lower_body_ids
    ann_info['joint_weights'] = dataset_info.joint_weights
    ann_info['skeleton'] = dataset_info.skeleton
    sigmas = dataset_info.sigmas
    dataset_name = dataset_info.dataset_name

    net = tompnet.tompnet101(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                            num_kpts=settings.num_kpts,
                            head_feat_norm=True, final_conv=True, out_feature_dim=256, feature_sz=settings.feature_sz,
                            frozen_backbone_layers=settings.frozen_backbone_layers,
                            num_encoder_layers=settings.num_encoder_layers,
                            num_decoder_layers=settings.num_decoder_layers,
                            use_test_frame_encoding=settings.use_test_frame_encoding,
                            cfg=settings)
    net = net.to(device)
    data = data.to(device)
    for k,v in data.items():
        if k.startswith('train') and type(v)==torch.Tensor:
            data[k] = v[:,None]
    lte = eut.LTREvaluator(net,settings,cfg,dif=dataset_info)
    # lte = eut_v1.LTREvaluator(net,settings,cfg,dif=dataset_info)
    xx= lte.generate_results_video(data,val_pipeline_,test_pipeline_,ann_info,cfd,label_params,dataset_info,device,padding=2)
