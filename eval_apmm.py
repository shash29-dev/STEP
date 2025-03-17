import random
import numpy as np
import pdb

import torch
from omegaconf import OmegaConf,open_dict
import torch.optim as optim
import src.utils.utilfunc as mut
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes
import os
import torchvision
from torchvision.utils import draw_keypoints
import time
from src.models.tracking import tompnet
from src.models.loss.bbr_loss import GIoULoss, KpMSELoss
import src.models.loss as ltr_losses
import src.actors.tracking as actors
from src.trainers.ltr_trainer import LTRTrainer
import json
import pandas as pd
import src.eval.evalutils_apmm as eut
import src.utils.visutils_eval as vut
from data.tensordict import TensorDict
from data.mmpipeline_trans import train_pipeline, val_pipeline
from mmpose.datasets.dataset_info import DatasetInfo
import importlib

def evaluation_loop(cfg):
    cfd = cfg.data
    settings= cfg.data.settings
    device= 'cuda'

    nkp =10
    annotjson = settings.annotjson
    with open(annotjson, "r") as f:
        annotations = json.load(f)
    annotations= mut.EasyDict(annotations)
    anno= pd.DataFrame(annotations.annotations)
    images= pd.DataFrame(annotations.images)

    ## APT36K has D:\\ starting
    if images.file_name[0].startswith('D:\\'):
        images['file_name'] = images['file_name'].str.replace(r'^D:\\Animal_pose\\AP-36k-patr1', '', regex=True)
        images['file_name'] = images['file_name'].str.replace(r'\\', '/', regex=True)

    ## Set this flag to False for other data
    clean = True # path in APT36K starts has some non-ascii vals
    if clean:
        for idx,fname in enumerate(images.file_name):
            xx= fname.split('\\')
            xxb= [x.isascii() for x in xx]
            if False in xxb:
                tmp =[]
                for txt,txtb in zip(xx,xxb):
                    if txtb==True:
                        tmp.append(txt)
                    else:
                        clean = "".join([x for x in txt if x.isascii()])
                        tmp.append(clean)
                modfname = "\\".join(tmp)
                print(modfname)
                images.at[idx,'file_name'] = modfname
                
    cat = pd.DataFrame(annotations.categories)
    
    if cfg.data.kind !='jrdbpose':
    ##### Added Val APT36K###
        vid_an =[]
        # anno = anno[anno.num_keypoints>nkp]
        seed_value =100  # Seed for evaluation purposes
        atleast_one_category = anno.groupby('video_id', group_keys=False) #.apply(lambda x: x.sample(5,random_state=seed_value))
        seq= []
        
        for idx,row in atleast_one_category:
            vid = np.unique(row.video_id)
            assert  len(np.unique(row.video_id)) == 1
            vid = vid[0]
            seq.append(anno[anno['video_id'] == vid])
    
    elif cfg.data.kind == 'jrdbpose':
        raise NotImplementedError
    
    val_set = pd.concat(seq, ignore_index=True)
    videos= pd.unique(val_set.video_id)
    num_track_ids = 0
    for vidid in videos:
        video = vidid
        vidanno= anno[anno.video_id==video]
        im= (images[images['id']==vidanno.image_id.iloc[0]].file_name).item() 
        
        # Filter Gorilla for eval
        if 'gorilla' not in im:
            continue
        vidanno = vidanno.sort_values(by="track_id")
        animals = pd.unique(vidanno.track_id)
        num_track_ids+= len(animals)
        for animal in animals:
            vid_an.append([video,animal])

        # Collect max 10 for visualization
        if len(vid_an)>10:
            break

    
    num_train_frames= 2
    all_result=[]
    eval_model = True
    if eval_model:
        with open_dict(settings):
            settings.output_sz = settings.feature_sz * 16     # Size of input image crop
        output_sigma = settings.output_sigma_factor / settings.search_area_factor
        label_params = {'feature_sz': settings.feature_sz, 
                        'sigma_factor': output_sigma, 
                        'kernel_sz': settings.target_filter_sz}
        pipeline= val_pipeline
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
        lte = eut.LTREvaluator(net,settings,cfg,dif=dataset_info)
        
        for index in range(len(vid_an)):
            video_id, track_id  = vid_an[index]
            vidanno= anno[anno.video_id==video_id]
            vidannoanimal = vidanno[vidanno.track_id==track_id]
            print('Processing..{}/{}\t num frames: {}'.format(index,len(vid_an),len(vidannoanimal)))
            if len(vidannoanimal)<4:
                continue
            afd= eut.get_frames_apmm(vidannoanimal,images,cat,cfd.kwargs.root,cfd.settings.num_kpts)
            afd= mut.EasyDict(afd)
            train_data = {}
            test_data = {}
            for key,value in afd.items():
                train_data[key] = value[:num_train_frames]
                test_data[key] = value[num_train_frames:]  # data_processing don't work if test_images>1. Use test_kpts for visualization OR Correct test_kpts_target, test_kpts_label.. Find issue.
                # test_data[key] = value[num_train_frames:num_train_frames+1]  # For Ex. Check this. It works with test_kpts_target, test_kpts_label.

            train_data = mut.EasyDict(train_data)
            test_data = mut.EasyDict(test_data)
            data = TensorDict (
                {
                    'train_images': train_data.all_images,
                    'train_bbox': train_data.all_bbox,
                    'train_skeleton': train_data.all_skeleton,
                    'train_kpts': train_data.all_kpts,
                    'train_impath': train_data.impath,

                    'test_images': test_data.all_images,
                    'test_bbox': test_data.all_bbox,
                    'test_skeleton': test_data.all_skeleton,
                    'test_kpts': test_data.all_kpts,
                    'test_impath': test_data.impath,
                }
            )
            xx= data.copy()
            data,metas, vd = mut.get_processed(data,pipeline,ann_info,cfd,label_params)
            data = data.to(device)
            for k,v in data.items():
                if k.startswith('train') and type(v)==torch.Tensor:
                    data[k] = v[:,None]
            xx= lte.generate_results(data,animid=index,metas=metas,vd=vd,video_id=video_id,track_id=track_id)
            # print(video_id,track_id,index,len(all_result))

            all_result.extend(xx)
            
        rfname= cfg.snaps.image_save_dir.replace('images','results')+'/{}.json'.format(cfg.data.fname)
        print('Saving...{}'.format(rfname))
        os.makedirs(os.path.dirname(rfname),exist_ok=True)
        with open(rfname,'w') as f: json.dump(all_result,f)
    
    rfname = './logs/vitpose/results_vit_apt36.json'
    rfname= cfg.snaps.image_save_dir.replace('images','results')+'/{}.json'.format(cfg.data.fname)
    # rfname = './logs/stepvit/snaps/results/aptmmpose/results_noroll_pd125.json'
    with open(rfname,'r') as f: all_result = json.load(f)
    dataset_info_path = cfd.settings.dataset_info
    dcname = dataset_info_path.split(os.sep)[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(dcname, dataset_info_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_info = getattr(module,dcname)
    dataset_info = DatasetInfo(dataset_info)
    vut.save_gif(pd.DataFrame(all_result),dataset_info,cfg)
