import json
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.trainers import BaseTrainer
import pdb
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes, draw_keypoints
import torchvision
import src.utils.utilfunc_awazi as mut
from data.tensordict import TensorDict
import glob
import imageio
from mmpose.core import imshow_bboxes, imshow_keypoints
import mmcv
from mmpose.datasets.dataset_info import DatasetInfo
import cv2
from data.tensordict import TensorDict
import pandas as pd
from src.eval.pred_track_utils import PredTrack, EvalUtils

class LTREvaluator():
    def __init__(self, net, settings,cfg,dif):
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.net = net
        self.training = False
        self.net.train(self.training)
        self.settings= settings
        self.cfg=cfg
        self.update_settings(settings)
        self.load_checkpoint()
        self.dif = dif
        self.awazi_fol = '/home/cvig/shashikant/PoseEstimation/AnimalPoseGit/Evaluation/awazi/'
        self.ignore_mask = 'src/eval/error_map3.png'
        self.eut = EvalUtils()
        
    def generate_results_video(self,jp,pipeline,test_pipeline,ann_info,cfd,label_params,dataset_info,device,padding=3):
        self.ptdata = PredTrack()
        torch.set_grad_enabled(self.training)
        with open(jp,'r') as f: datajson_ = json.load(f)
        # datajson = datajson_['coco_data_all']
        datajson = datajson_
        dict_keys =[int(x) for x in list(datajson.keys())]
        trck_id_counter= 0
        self.ptdata.gtdata = datajson
        self.ptdata.save_dir = self.cfg.snaps.image_save_dir+'/awazi'
        os.makedirs(self.ptdata.save_dir,exist_ok=True)
        self.ptdata.savefname = os.path.basename(jp)
        
        bbox_offset = 0.0
        # 500
        sfri = 0
        for idx in range(len(dict_keys)):
            idx +=sfri
            # print('For this json path Currently Running.... {}/{}'.format(idx, len(dict_keys)))
            val = datajson[str(dict_keys[idx])]
            train_frame = val['file_name']
            bboxes = val['bboxes']
            trfid = val['frame_id']
            train_frame = train_frame.replace('/home/cvig/shashikant/', '/media/shashikant/A/shashikant/')
            # train_image = plt.imread(self.awazi_fol+train_frame)
            train_image = plt.imread(train_frame)
            # mask = cv2.imread(self.ignore_mask)
            # mask = cv2.resize(mask,(train_image.shape[1],train_image.shape[0]))
            # if len(bboxes)==4:
            #     bboxes = [bboxes[0],bboxes[1]]
            self.ptdata.add_predictions(None,None,trfid,None,only_frame=True)
            for bboxenum, bbox_in in enumerate(bboxes):
                # bbox_in = self.eut.add_offset_to_bbox(bbox_in,bbox_offset)
                # x, y, w, h = map(int, bbox_in)
                # roi = mask[y:y+h, x:x+w]
                # if roi.sum()>1000:
                #     # print('In ignore region... Continuing')
                #     continue
                if trfid in self.ptdata.preds.keys():
                    bboxes_pd = self.ptdata.preds[trfid]['bbox']
                    istracked,all_iou,bbpd = self.ptdata.is_frame_tracked(bboxes,bboxes_pd,thresh=0.1)
                    # if istracked:
                    #     bbpd = [self.eut.add_offset_to_bbox(bbpd[0],bbox_offset)]
                    #     self.ptdata.preds[trfid]['bbox'] = bbpd
                    if istracked:
                        print('BBox already Tracked.. Continuing')
                        continue
                    else:
                        trck_id_counter += 1
                        
                data = TensorDict({
                    'train_images': [train_image,train_image],
                    'train_bbox': [np.array(bbox_in),np.array(bbox_in)]
                })
                datain = data.to(device) 
                all_im=[]
                adfl = []
                datas1={}
                data={}
                num_train_frames = len(datain['train_images'])
                # num_train_frames = 1
                padding=3
                for idxt in range(idx+1,len(dict_keys)):
                    print('{}\t For frame {}/{} Tracking Box  {}/{} ... {}/{}'.format(os.path.basename(jp),idx,len(dict_keys),bboxenum,len(bboxes),idxt-idx,len(dict_keys)),end='\r')
                    val = datajson[str(dict_keys[idxt])]
                    test_frame = val['file_name']
                    test_frame = test_frame.replace('/home/cvig/shashikant/', '/media/shashikant/A/shashikant/')
                    # tim = plt.imread(self.awazi_fol+test_frame)
                    tim = plt.imread(test_frame)
                    # Process Stage-1
                    datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding=3)
                    # data = {}  # Resets added kpts labels, ltrb
                    datas1['train_images'] = datatrain['train_images'][:,None]
                    datas1['train_bbox'] = datatrain['train_bbox'][:,None]
                    datas1['train_label'] = datatrain['train_label'][:,None]
                    datas1['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
                    datas1['train_skeleton'] = datatrain['train_skeleton'][:,None]
                    gtflag=0
                    if len(val['bboxes'])!=0:
                        datatest,_ = mut.get_processed_traininfo([tim],[np.array(val['bboxes'][bboxenum])],pipeline,ann_info,cfd,label_params,dataset_info,padding=1.5,key_label='test_')
                        gtflag=1
                    else:
                        datatest= mut.get_processed_testframe_giventraininfo([tim], datatrain ,test_pipeline,ann_info,cfd,label_params,dataset_info)
                    datas1['test_images'] = datatest['test_images'][:,None]
                    datas1 = TensorDict(datas1)
                    datas1 = datas1.to(device)
                    target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=datas1['train_images'],
                                                test_imgs=datas1['test_images'],
                                                train_bb=datas1['train_bbox'],
                                                train_label=datas1['train_label'],
                                                train_kpts_label=None,
                                                train_ltrb_target=datas1['train_ltrb_target'],
                                                train_kpts_target=None,
                                                train_target = 'gmsp',
                                                )
                    
                    target_scores = torch.sigmoid(target_scores)
                    kpim ,bbox,kpts, bboxorims1, kptsorim, _= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, datas1,imname='tmp.png',metas=datatrain['train_img_metas'][-1],image_orig=tim)
                    
                    # Process: Stage 2
                    # data= {}
                    datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding=2)
                    data['train_images'] = datatrain['train_images'][:,None]
                    data['train_bbox'] = datatrain['train_bbox'][:,None]
                    data['train_label'] = datatrain['train_label'][:,None]
                    data['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
                    data['train_skeleton'] = datatrain['train_skeleton'][:,None]
                    if len(val['bboxes'])!=0:
                        datatest,_ = mut.get_processed_traininfo([tim],[np.array(val['bboxes'][bboxenum])],pipeline,ann_info,cfd,label_params,dataset_info,padding=padding,key_label='test_')
                    else:
                        datatest,_= mut.get_processed_traininfo([tim], [bboxorims1[0]] ,pipeline,ann_info,cfd,label_params,dataset_info,key_label='test_',padding=1.5)
                    data['test_images'] = datatest['test_images'][:,None]
                    data = TensorDict(data)
                    data = data.to(device)
                    
                    target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_bb=data['train_bbox'],
                                                train_label=data['train_label'],
                                                train_kpts_label=None,
                                                train_ltrb_target=data['train_ltrb_target'],
                                                train_kpts_target=None,
                                                train_target = 'gmsp',
                                                )
                    target_scores = torch.sigmoid(target_scores)
                    kpim ,bbox,kpts, bboxorim, kptsorim, addframe= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, data,imname='tmp.png',metas=datatest['test_img_metas'][-1],image_orig=tim)
                    
                    istracked,_,bbpd = self.ptdata.is_frame_tracked(val['bboxes'],[bboxorim[0].tolist()],thresh=0.2)
                    # bbpd = [add_offset_to_bbox(bbpd[0],bbox_offset)]
                    bboxorim = np.array(bbpd)
                    
                    # all_im.append(kpim)
                    if idxt-idx<40: addframe=True
                    if idxt-idx>5: adfl.append(addframe)
                    # addframe=False
                    if gtflag==1:
                        addframe=True
                        gtflag=0
                    addframe = True
                    if addframe:
                        print('Added Frame')
                        datain['train_images'].append(tim)
                        datain['train_bbox'].append(bboxorim[0])
                        data['train_images'] = torch.cat((data['train_images'],data['test_images']),dim=0)
                        data['train_bbox']=torch.cat((data['train_bbox'], bbox[None,None]),dim=0)
                        data['train_label']=torch.cat((data['train_label'],vd._generate_label_function(data['train_bbox'][-1])[None]),dim=0)
                        tltrb, tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                        data['train_ltrb_target']=torch.cat((data['train_ltrb_target'],tltrb[None]),dim=0)
                        
                        datas1['train_images'] = torch.cat((datas1['train_images'],datas1['test_images']),dim=0)
                        datas1['train_bbox']=torch.cat((datas1['train_bbox'], bbox[None,None]),dim=0)
                        datas1['train_label']=torch.cat((datas1['train_label'],vd._generate_label_function(datas1['train_bbox'][-1])[None]),dim=0)
                        tltrb, tsr = vd._generate_ltbr_regression_targets(datas1['train_bbox'][-1])
                        datas1['train_ltrb_target']=torch.cat((datas1['train_ltrb_target'],tltrb[None]),dim=0)
                        
                        slide = True
                        if slide:
                            # datain['train_images'] =   datain['train_images'][:1] + datain['train_images'][-num_train_frames:]
                            # datain['train_bbox'] =     datain['train_bbox'][:1] + datain['train_bbox'][-num_train_frames:]
                            datain['train_images'] =   datain['train_images'][-num_train_frames:]
                            datain['train_bbox'] =     datain['train_bbox'][-num_train_frames:]
                            
                        
                       
                    # ncf=5
                    # numfrth=1500
                    # has_n_consecutive_false = any(all(adfl[i + j] is False for j in range(ncf)) for i in range(len(adfl) - ncf + 1))
                    # if has_n_consecutive_false:
                    #     print('Pruned Tracking... {} consecutive Triggers'.format(ncf))
                    #     break
                    # if idxt-idx> numfrth:
                    #     print('Pruned Tracking... {} frames threshold exceeded'.format(numfrth))
                    #     break
                    
                    prev_frameid = datajson[str(dict_keys[idxt-1])]['frame_id']
                    if prev_frameid in self.ptdata.preds.keys():
                        if len(self.ptdata.preds[prev_frameid]['bbox'])>0:
                            prev_box = self.ptdata.preds[prev_frameid]['bbox'][-1]
                            prev_tid =  self.ptdata.preds[prev_frameid]['track_id'][-1]
                            if prev_tid==trck_id_counter:
                                iou = self.calculate_iou(prev_box,bboxorim[0].tolist())
                                distance = self.calculate_distance_between_centers(prev_box,bboxorim[0])
                                # if iou<0.001 and distance>prev_box[2]*45:
                                if  distance>80:
                                    print('Pruned Tracking... {}/{} Very Less IoU'.format(iou,distance))
                                    # break
                    self.ptdata.add_predictions(bboxorim[0].tolist(),kptsorim[0].flatten().tolist(),val['frame_id'],trck_id_counter)
                    
                
                trck_id_counter += 1
                # output_dir =  self.cfg.snaps.image_save_dir
                # gifname = 'animal_{}.mp4'.format(0)
                # gifname = os.path.join(output_dir,gifname)
                
                # if len(all_im)>0:
                #     if type(all_im[0])==torch.Tensor:
                #         all_im = [(im.permute(1,2,0).numpy()*255).astype(np.uint8) for im in all_im]
                
                # imageio.mimsave(gifname,all_im[5:])
                # time.sleep(3)
            #     if idx>sfri+5:
            #         break
            # if idx>sfri+5:
            #     break
        self.ptdata.save_jsons()
                
    @staticmethod  
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou
    
    @staticmethod
    def calculate_distance_between_centers(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center_x1, center_y1 = x1 + w1 / 2, y1 + h1 / 2
        center_x2, center_y2 = x2 + w2 / 2, y2 + h2 / 2

        distance = math.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

        return distance

    def visualize_all_eval(self,scores,bbop,kpop, target_kpts_scores, data, imname,metas,image_orig):
        ns = scores.shape[1]
        nf = scores.shape[0]
        # tlb = data['test_label']
        idsp = scores.reshape(nf, ns, -1).max(dim=2)[1]
        score_pdbb = scores.reshape(nf,ns,-1).max(dim=2)[0].item() 
        
        tfs = self.settings.feature_sz
        stride = 16
        train_img_sample_sz = torch.Tensor([tfs*stride, tfs*stride]).to(scores.device)

        shifts_x = torch.arange(0, train_img_sample_sz[0], step=16,dtype=torch.float32).to(scores.device)
        shifts_y = torch.arange(0, train_img_sample_sz[1], step=16,dtype=torch.float32).to(scores.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 16 // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores.shape[2:]
        xs = xs.reshape(s1, s2).repeat(nf,ns,1,1)
        ys = ys.reshape(s1, s2).repeat(nf,ns,1,1)

        ltrb = bbop.permute(0,1,3,4,2) * train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrb[:, :, :,:, 0]
        xs2 = xs + ltrb[:, :,:,:, 2]
        ys1 = ys - ltrb[:, :,:,:, 1]
        ys2 = ys + ltrb[:, :, :,:,3]
        xy = torch.stack((xs1,ys1,xs2,ys2),axis=2)
        xym = xy.reshape(nf,ns,4,-1)[0, torch.arange(0, ns), :, idsp].view(nf, ns, 4)
        xym[:,:,2:] = xym[:,:,2:] - xym[:,:,:2]
        predbb_xywh= xym

        
        output_dir =  self.cfg.snaps.image_save_dir
        name = imname

        
        orim = image_orig
        center = metas['center']
        scale = metas['scale']
        # scale = scale * 200.0
        scale = scale * 200
        test_images = data['test_images']
        # unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        # xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        # img = xx[0].permute(1,2,0).cpu().numpy()
        output_size = data['train_images'].shape[-2:]
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
        
        # bbpred --> recon from ltrb
        bbpred = predbb_xywh.view(-1,predbb_xywh.shape[-1])
        bboxes_pred = bbpred[0]
        bboxes = torchvision.ops.box_convert(bbpred[0][None], in_fmt='xywh', out_fmt='xyxy').cpu().numpy().astype(int)
        bboxes = bboxes.reshape(-1,2)
        
        # bboxes =  data['train_bbox'][0][0].cpu().numpy().reshape(-1,2).astype(int)
        
        target_coords = np.ones_like(bboxes)
        target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        bboxes_orim = target_coords.flatten()[None]
        bboxes_orim_xyxy = torchvision.ops.box_convert(torch.Tensor(bboxes_orim)[0], in_fmt='xyxy', out_fmt='xywh').cpu().numpy().astype(int)
       
       
        ## kpts --> preds
        nkp = kpop.shape[2]
        kpops = kpop.permute(0,1,3,4,2) * train_img_sample_sz[[0, 1]*(nkp//2)]
        kpops = kpops.reshape(*kpops.shape[:-1],nkp//2,-1)
        kpx = [xs - kpops[...,idx,0] for idx in range(nkp//2)]
        kpy = [ys - kpops[...,idx,1] for idx in range(nkp//2)]
        kpxy=[]
        for lx,ty in zip(kpx,kpy):
            kpxy.append(lx)
            kpxy.append(ty)
        kp = torch.stack(kpxy, dim=2)
        tlbkpt = torch.stack(target_kpts_scores,axis=2)
        idspdkp = [x.reshape(nf,ns,-1).max(dim=2)[1] for x in  tlbkpt.permute(2,0,1,3,4)]

        #TODO Refine based on score here?
        score_pdkp = np.array([x.reshape(nf,ns,-1).max(dim=2)[0].item() for x in  tlbkpt.permute(2,0,1,3,4)])
        # print(score_pdkp.min())
        thresh= 0.7
        confidence = score_pdkp>thresh
        # if confidence.sum()> len(confidence)-15:
        # if score_pdkp.min()>thresh and score_pdbb>thresh and confidence.sum()> len(confidence)-8:
        # if score_pdkp.min()>thresh and confidence.sum()> len(confidence)-4:
        if score_pdbb>thresh:
            addframe=True
        else:
            # print('Low Confidence Frame............................................ {}'.format(score_pdbb))
            addframe=False
        
        color_thresh =0.50
        if (score_pdkp>color_thresh).sum() > len(confidence)-2: # int: how many can be wrong
            color='green'
        else:
            color='blue'
        
        # bpo=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors=color,thickness=2,show=False)
        # # bpo_=mmcv.imshow_bboxes(img.copy(),bboxes.flatten()[None],colors='green',thickness=2,show=False)
        # plt.imsave(os.path.join(output_dir,name),bpo)
        # pdb.set_trace()
        
        kpm_ = [kp.reshape(nf,ns,nkp,-1)[0, torch.arange(0, ns), :, x].view(nf, ns, nkp) for x in idspdkp]
        kpm_ = [x.reshape(*x.shape[:-1],nkp//2,-1) for x in kpm_]
        fkp =[]
        for idx in range(len(kpm_)):
            fkp.append(kpm_[idx][:,:,idx,:])
        fkp = torch.stack(fkp,axis=2)

        kp = fkp.view(-1,*fkp.shape[2:])
        conn = data['train_skeleton'][0][0].long().tolist()
        conn= [conn]*kp.shape[0]
        vis= torch.ones((kp.shape[0],kp.shape[1],1)).to(scores.device)
        vis[kp.sum(axis=-1)<=0]=0
        # vis[:,~confidence,:]=0
        kpv = torch.cat((kp,vis),axis=-1)
        kpv_pred= kpv
        skeleton = np.tile(np.array(self.dif.skeleton), (kpv.shape[1],1,1))
        pose_kpt_color = np.tile(self.dif.pose_kpt_color, (kpv.shape[1],1,1))
        pose_link_color = np.tile(self.dif.pose_link_color, (kpv.shape[1],1,1))

        kpv= kpv.cpu().numpy()
        # kpp = imshow_keypoints(img.copy(),kpv,skeleton=skeleton[0],
        #                       pose_kpt_color=pose_kpt_color[0],
        #                       pose_link_color=pose_link_color[0], 
        #                       radius=3,
        #                       thickness=2)
        # plt.imsave(os.path.join(output_dir,name), kpp)
        # pdb.set_trace()

        target_coords = np.ones_like(kpv[0])
        target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
        target_coords[:, 2] = kpv[0][:, 2] 
        # connvis= []
        # conncolorvis=[]
        # for idx,(sk,kpsk,bb,plc) in enumerate(zip(conn,kp,bbpred,pose_link_color)):
        #     connsub=[]
        #     conncolor=[]
        #     for ske,plce in zip(sk,plc):
        #         sp = kpsk[ske[0]] 
        #         ep = kpsk[ske[1]] 
        #         if sp.sum()<=10:
        #             continue
        #         if ep.sum()<=10:
        #             continue
        #         bbox_x, bbox_y, bbox_w, bbox_h = bb.tolist()
        #         offset = 5
        #         bbox_x, bbox_y, bbox_w, bbox_h = (bbox_x - offset, bbox_y - offset, bbox_w + 2 * offset, bbox_h + 2 * offset)
        #         inside_ep = bbox_x <= ep[0].item() <= bbox_x + bbox_w and bbox_y <= ep[1].item() <= bbox_y + bbox_h
        #         inside_sp = bbox_x <= sp[0].item() <= bbox_x + bbox_w and bbox_y <= sp[1].item() <= bbox_y + bbox_h
        #         if inside_ep and inside_sp:
        #             connsub.append(ske)
        #             conncolor.append(plce)
        #     connvis.append(connsub)
        #     conncolorvis.append(conncolor)

        # skel = np.array(connvis[0])
        # plc= np.array(conncolorvis[0])
        # kppo = imshow_keypoints(bpo,target_coords[None],skeleton=skel,
        #                       pose_kpt_color=pose_kpt_color[0],
        #                       pose_link_color=plc, 
        #                       radius=10,
        #                       thickness=5)
        
        ## kpts --> recon from gt
        # kgto= None
        kpim = None
        # kpim = np.hstack((kgto,kppo)) if kgto is not None else kppo
        # print('\n............Save a Sample.........\n')
        # plt.imsave(os.path.join(output_dir,name), kpim)
        target_coords[:,-1]=score_pdkp
        return kpim, bboxes_pred, kpv_pred, bboxes_orim_xyxy[None], target_coords[None], addframe
    
    

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.net
        net_type = type(net).__name__
        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._checkpoint_dir, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, 
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        # checkpoint_dict = loading.torch_load_legacy(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()

        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net_keys = net.state_dict().keys()
                loaded_keys = checkpoint_dict[key].keys()
                weights= {}
                for nkey in net_keys:
                    if nkey in loaded_keys:
                        weights[nkey]=checkpoint_dict[key][nkey]
                    else:
                        weights[nkey]=net.state_dict()[nkey]
                        print('Skipping Loading... Key in Network Exist and Not found in Loaded: {}'.format(nkey))
                net.load_state_dict(weights)
                
            elif key == 'optimizer':
                try:
                    self.optimizer.load_state_dict(checkpoint_dict[key])
                except:
                    print('Couldnt Load Optimizer States')
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # # Update the epoch in lr scheduler
        # if 'epoch' in fields:
        #     self.lr_scheduler.last_epoch = self.epoch

        print('Loaded Model...')
        return True
