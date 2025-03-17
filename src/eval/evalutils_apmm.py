import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.trainers import BaseTrainer
import pdb
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes, draw_keypoints
import torchvision
import src.utils.utilfunc as mut
from data.tensordict import TensorDict
import glob
import imageio
from mmpose.core import imshow_bboxes, imshow_keypoints
import mmcv
from mmpose.datasets.dataset_info import DatasetInfo


def get_frames(frames,images,cat,data_path,num_kpts):
    all_frames_dict= {
        'all_images': [],
        'all_bbox': [],
        'all_skeleton': [],
        'all_kpts': [],
        'impath':[],
    }
    for idx,data in frames.iterrows():
        imid = data.image_id
        im= (images[images['id']==imid].file_name).item() 
        #  [x.isascii() for x in xx]
        catanno = cat[cat.id == data.category_id]
        fpath_ = os.sep.join(im.split(data_path.split(os.sep)[-1])[-1].split('\\'))
        if fpath_.startswith('/'): fpath_= fpath_[1:]
        impath = data_path + os.sep + fpath_


        bbox = np.array([x for x in data.bbox])
        pdb.set_trace()
        kpts = np.array(data.keypoints).reshape(-1,3)[:,:2]
        kpts = kpts[:num_kpts]
        skeleton = np.array(catanno.skeleton.item())-1
        skeleton = skeleton.astype(int)
        image = plt.imread(impath)

        all_frames_dict['impath'].append(impath)
        all_frames_dict['all_images'].append(image)
        all_frames_dict['all_bbox'].append(torch.Tensor(bbox))
        all_frames_dict['all_skeleton'].append(torch.Tensor(skeleton))
        all_frames_dict['all_kpts'].append(torch.Tensor(kpts))
    
    # all_frames_dict['all_images']= torch.from_numpy(np.array(all_frames_dict['all_images'])).permute(0,3,1,2)
    # all_frames_dict['all_bbox']= torch.from_numpy(np.array(all_frames_dict['all_bbox']))
    # all_frames_dict['all_skeleton']= torch.from_numpy(np.array(all_frames_dict['all_skeleton']))
    # all_frames_dict['all_kpts']= torch.from_numpy(np.array(all_frames_dict['all_kpts']))
    return all_frames_dict


def get_frames_apmm(frames,images,cat,data_path,num_kpts):
    all_frames_dict= {
        'all_images': [],
        'all_bbox': [],
        'all_skeleton': [],
        'all_kpts': [],
        'impath':[],
    }
    for idx,data in frames.iterrows():
        imid = data.image_id
        im= (images[images['id']==imid].file_name).item() 
        #  [x.isascii() for x in xx]
        catanno = cat[cat.id == data.category_id]
        impath = data_path+im

        bbox = np.array([x for x in data.bbox])
        kpts = np.array(data.keypoints).reshape(-1,3)[:,:2]
        kpts = kpts[:num_kpts]
        skeleton = np.array(catanno.skeleton.item())-1
        skeleton = skeleton.astype(int)
        image = plt.imread(impath)

        all_frames_dict['impath'].append(impath)
        all_frames_dict['all_images'].append(image)
        all_frames_dict['all_bbox'].append(torch.Tensor(bbox))
        all_frames_dict['all_skeleton'].append(torch.Tensor(skeleton))
        all_frames_dict['all_kpts'].append(torch.Tensor(kpts))
    
    # all_frames_dict['all_images']= torch.from_numpy(np.array(all_frames_dict['all_images'])).permute(0,3,1,2)
    # all_frames_dict['all_bbox']= torch.from_numpy(np.array(all_frames_dict['all_bbox']))
    # all_frames_dict['all_skeleton']= torch.from_numpy(np.array(all_frames_dict['all_skeleton']))
    # all_frames_dict['all_kpts']= torch.from_numpy(np.array(all_frames_dict['all_kpts']))
    return all_frames_dict


def get_syn_frames(frames,data_path):
    all_frames_dict= {
        'all_images': [],
        'all_bbox': [],
        'all_skeleton': [],
        'all_kpts': []
    }
    for idx,data in frames.iterrows():
        imid = data.image_id
        imname = str(imid).zfill(5) + '.jpg'
        camera = 'Camera{}'.format(data.video_id)
        impath = data_path + os.sep+ camera + os.sep + 'Target' + os.sep + imname
        bbox = np.array([x for x in data.bbox])
        kpts = np.array(data.keypoints).reshape(-1,3)[:,:2]
        kpts = kpts
        skeleton = np.zeros((17,2))
        skeleton = skeleton.astype(int)
        image = plt.imread(impath)

        all_frames_dict['all_images'].append(image)
        all_frames_dict['all_bbox'].append(torch.Tensor(bbox))
        all_frames_dict['all_skeleton'].append(torch.Tensor(skeleton))
        all_frames_dict['all_kpts'].append(torch.Tensor(kpts))
    
    # all_frames_dict['all_images']= torch.from_numpy(np.array(all_frames_dict['all_images'])).permute(0,3,1,2)
    # all_frames_dict['all_bbox']= torch.from_numpy(np.array(all_frames_dict['all_bbox']))
    # all_frames_dict['all_skeleton']= torch.from_numpy(np.array(all_frames_dict['all_skeleton']))
    # all_frames_dict['all_kpts']= torch.from_numpy(np.array(all_frames_dict['all_kpts']))
    return all_frames_dict


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

    def generate_results(self,data, animid=0,metas=None,vd=None,video_id=0,track_id=0):
        torch.set_grad_enabled(self.training)
        all_im =[]
        fresult= []
        num_train_frames = len(data['train_images'])
        for idx,tim in enumerate(data['test_images']):
            target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=data['train_images'],
                                                test_imgs=tim[None,None],
                                                train_bb=data['train_bbox'],
                                                train_label=data['train_label'],
                                                train_kpts_label=None,
                                                train_ltrb_target=data['train_ltrb_target'],
                                                train_kpts_target= None,
                                                train_target='gmsp'
                                                )
            data_tmp = TensorDict (
                        {
                        'train_images': data['train_images'],
                        'train_bbox': data['train_bbox'],
                        'train_skeleton': data['train_skeleton'],
                        'train_kpts': data['train_kpts'],
                        'test_images': tim[None,None],
                        'test_label': data['test_label'][idx][None,None],
                        'test_ltrb_target' : data['test_ltrb_target'][idx][None,None],
                        'test_bbox' : data['test_bbox'][idx][None,None],
                        'test_kpts' : data['test_kpts'][idx][None,None],
                        }
                    )   
            kpim ,bbox,kpts, result= self.visualize_all_or(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png',metas=metas[idx])
            result['orim_bbox']=result['orim_bbox'][0].tolist()
            result['orim_kpts']=result['orim_kpts'].flatten().tolist()
            result['orim_bbox_gt'] = result['orim_bbox_gt'][0].tolist()
            result['orim_kpts_gt'] = result['orim_kpts_gt'].flatten().tolist()
            result['image_file'] = metas[idx]['image_file']
            result['video_id'] = int(video_id)
            result['track_id'] = int(track_id)
            
            fresult.append(result)
            # Update memory --> Note update with xywh box Not xyxy
            slide = True
            if slide:
                data['train_images'] = torch.cat((data['train_images'],tim[None,None]),dim=0)
                data['train_bbox']=torch.cat((data['train_bbox'], data['test_bbox'][:][idx][None,None]),dim=0)

                data['train_label']=torch.cat((data['train_label'],vd._generate_label_function(data['train_bbox'][-1])[None]),dim=0)
                tltrb, tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target']=torch.cat((data['train_ltrb_target'],tltrb[None]),dim=0)
                data['train_images']=               data['train_images'][-num_train_frames:]
                data['train_bbox']=                 data['train_bbox'][-num_train_frames:]
                data['train_label']=                data['train_label'][-num_train_frames:]
                data['train_ltrb_target']=          data['train_ltrb_target'][-num_train_frames:]
            
        return fresult

    def visualize_all_or(self,scores,bbop,kpop, target_kpts_scores, data, imname,metas):
        ns = scores.shape[1]
        nf = scores.shape[0]
        tlb = data['test_label']
        idsp = scores.reshape(nf, ns, -1).max(dim=2)[1]
        idsgt = tlb.reshape(nf,ns,-1).max(dim=2)[1]
        # max_score, max_disp = dcf.max2d(scores)
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

        ltrbgt = data['test_ltrb_target'].permute(0,1,3,4,2) * train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrbgt[:, :, :,:, 0]
        xs2 = xs + ltrbgt[:, :,:,:, 2]
        ys1 = ys - ltrbgt[:, :,:,:, 1]
        ys2 = ys + ltrbgt[:, :, :,:,3]
        xy = torch.stack((xs1,ys1,xs2,ys2),axis=2)
        xym = xy.reshape(nf,ns,4,-1)[0, torch.arange(0, ns), :, idsgt].view(nf, ns, 4)
        xym[:,:,2:] = xym[:,:,2:] - xym[:,:,:2]
        gtbb_xywh= xym

        output_dir =  self.cfg.snaps.image_save_dir
        name = imname

        # bbgt --> recon from ltrb
        bbgt = gtbb_xywh.view(-1,gtbb_xywh.shape[-1])
        test_images = data['test_images']
        test_bbox = data['test_bbox']
        unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        bboxes_gt = bbgt[0]
        bboxes = torchvision.ops.box_convert(bbgt[0][None], in_fmt='xywh', out_fmt='xyxy').cpu().numpy().astype(int)
        img = xx[0].permute(1,2,0).cpu().numpy()
        # bgt=mmcv.imshow_bboxes(img,bboxes,colors='green',thickness=2,show=False)
        
        orim = plt.imread(metas['image_file'])
        center = metas['center']
        scale = metas['scale']
        scale = scale * 200.0
        output_size = img.shape
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
        bboxes = bboxes.reshape(-1,2)
        
        target_coords = np.ones_like(bboxes)
        target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        bboxes_orim = target_coords.flatten()[None]
        bboxes_gt =bboxes_orim.copy()
        
        # bgto=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors='magenta',thickness=8,show=False)
        # plt.imsave(os.path.join(output_dir,name), bgto)
        # bbpred --> recon from ltrb
        bbpred = predbb_xywh.view(-1,predbb_xywh.shape[-1])
        bboxes_pred = bbpred[0]
        bboxes = torchvision.ops.box_convert(bbpred[0][None], in_fmt='xywh', out_fmt='xyxy').cpu().numpy().astype(int)
        bboxes = bboxes.reshape(-1,2)
        target_coords = np.ones_like(bboxes)
        target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        bboxes_orim = target_coords.flatten()[None].copy()
        # bpo=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors='green',thickness=2,show=False)

        
        
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
        thresh= 0.50
        confidence = score_pdkp>thresh

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
        vis[:,~confidence,:]=0
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
        orim_kpts = target_coords.copy()
        # kppo = imshow_keypoints(bpo,target_coords[None],skeleton=skel,
        #                       pose_kpt_color=pose_kpt_color[0],
        #                       pose_link_color=plc, 
        #                       radius=10,
        #                       thickness=5)
        
        ## kpts --> recon from gt
        kpor = data['test_kpts']
        kp = kpor.view(-1,*kpor.shape[2:])
        conn = data['train_skeleton'][0][0].long().tolist()
        conn= [conn]*kp.shape[0]
        vis= torch.ones((kp.shape[0],kp.shape[1],1)).to(scores.device)
        vis[kp.sum(axis=-1)==0]=0
        # conn=conn[kp.sum(axis=-1)==0]
        kpv = torch.cat((kp,vis),axis=-1)
        kpv= kpv.cpu().numpy()
        # kgt = imshow_keypoints(img.copy(),kpv,skeleton=skeleton[0],
        #                       pose_kpt_color=pose_kpt_color[0],
        #                       pose_link_color=pose_link_color[0], 
        #                       radius=3,
        #                       thickness=2)

        target_coords = np.ones_like(kpv[0])
        target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
        target_coords[:, 2] = kpv[0][:, 2] 
        orim_kpts_gt = target_coords.copy()

        # kgto = imshow_keypoints(bgto,target_coords[None],skeleton=skeleton[0],
        #                       pose_kpt_color=pose_kpt_color[0],
        #                       pose_link_color=pose_link_color[0], 
        #                       radius=10,
        #                       thickness=5)
        kpim=None
        # kpim = np.hstack((kgto,kppo))
        # kpim = bgto
        # print('\n............Save a Sample.........\n')
        # plt.imsave(os.path.join(output_dir,name), kpim)
        # plt.imsave('tmp.png', kpim)
        
        dictionary_result = {
            'orim_bbox': bboxes_orim,
            'orim_kpts': orim_kpts,
            'orim_bbox_gt': bboxes_gt,
            'orim_kpts_gt': orim_kpts_gt,
        }
        # pdb.set_trace()
        # vm.vismap(img, orim.copy(), scores,bbop,kpop,target_kpts_scores, scale_x,scale_y,scale,center)
        return kpim, bboxes_pred, kpv_pred, dictionary_result
    
    def visualize_target(self,scores,bbop,kpop, target_kpts_scores, data, imname):
        ns = scores.shape[1]
        nf = scores.shape[0]
        tlb = data['test_label']
        idsp = scores.reshape(nf, ns, -1).max(dim=2)[1]
        idsgt = tlb.reshape(nf,ns,-1).max(dim=2)[1]
        # max_score, max_disp = dcf.max2d(scores)
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

        ltrbgt = data['test_ltrb_target'].permute(0,1,3,4,2) * train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrbgt[:, :, :,:, 0]
        xs2 = xs + ltrbgt[:, :,:,:, 2]
        ys1 = ys - ltrbgt[:, :,:,:, 1]
        ys2 = ys + ltrbgt[:, :, :,:,3]
        xy = torch.stack((xs1,ys1,xs2,ys2),axis=2)
        xym = xy.reshape(nf,ns,4,-1)[0, torch.arange(0, ns), :, idsgt].view(nf, ns, 4)
        xym[:,:,2:] = xym[:,:,2:] - xym[:,:,:2]
        gtbb_xywh= xym

        output_dir =  self.cfg.snaps.image_save_dir
        name = imname

        # bbgt --> recon from ltrb
        bbgt = gtbb_xywh.view(-1,gtbb_xywh.shape[-1])
        test_images = data['test_images']
        test_bbox = data['test_bbox']
        unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        xy_r = [draw_bounding_boxes(im, torchvision.ops.box_convert(box[None], in_fmt='xywh', \
                                                                      out_fmt='xyxy'), colors='red', width=4) \
                                                                      for im,box in zip(xx,bbgt)]
        
        # bbgt --> recon from bbox
        bbgt = test_bbox.view(-1,test_bbox.shape[-1])
        xy_rg = [draw_bounding_boxes(im, torchvision.ops.box_convert(box[None], in_fmt='xywh', \
                                                                      out_fmt='xyxy'), colors='green', width=2) \
                                                                      for im,box in zip(xy_r,bbgt)]
        
        ## preds
        bbpred = predbb_xywh.view(-1,predbb_xywh.shape[-1])
        xy_b = [draw_bounding_boxes(im, torchvision.ops.box_convert(box[None], in_fmt='xywh', \
                                                                      out_fmt='xyxy'), colors='blue', width=6) \
                                                                      for im,box in zip(xx,bbpred)]
        
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
        connvis= []
        kpv = torch.cat((kp,vis),axis=-1)
        for idx,(sk,kpsk,bb) in enumerate(zip(conn,kp,bbpred)):
            connsub=[]
            for ske in sk:
                sp = kpsk[ske[0]] 
                ep = kpsk[ske[1]] 
                if sp.sum()<=10:
                    continue
                if ep.sum()<=10:
                    continue
                bbox_x, bbox_y, bbox_w, bbox_h = bb.tolist()

                inside_ep = bbox_x <= ep[0].item() <= bbox_x + bbox_w and bbox_y <= ep[1].item() <= bbox_y + bbox_h
                inside_sp = bbox_x <= sp[0].item() <= bbox_x + bbox_w and bbox_y <= sp[1].item() <= bbox_y + bbox_h
                if inside_ep and inside_sp:
                    connsub.append(ske)
            connvis.append(connsub)

        imkpp_b = [draw_keypoints(im, kpim[None], colors="blue", connectivity=sk,radius=8, width=8) for im,kpim,sk in zip(xy_b,kpv,connvis)]
        
        ## kpts --> recon from test_kpts_target
        kpgtop= data['test_kpts_target']
        nkp = kpgtop.shape[2]
        kpgtops = kpgtop.permute(0,1,3,4,2) * train_img_sample_sz[[0, 1]*(nkp//2)]
        kpgtops = kpgtops.reshape(*kpgtops.shape[:-1],nkp//2,-1)
        kpx = [xs - kpgtops[...,idx,0] for idx in range(nkp//2)]
        kpy = [ys - kpgtops[...,idx,1] for idx in range(nkp//2)]
        
        kpxy=[]
        for lx,ty in zip(kpx,kpy):
            kpxy.append(lx)
            kpxy.append(ty)
        kp = torch.stack(kpxy, dim=2)
        tlbkpt = data['test_kpts_label']
        idsgtkp = [x.reshape(nf,ns,-1).max(dim=2)[1] for x in  tlbkpt.permute(2,0,1,3,4)]
        kpm_ = [kp.reshape(nf,ns,nkp,-1)[0, torch.arange(0, ns), :, x].view(nf, ns, nkp) for x in idsgtkp]
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
        kpv = torch.cat((kp,vis),axis=-1)
        connvis= []
        for sk,kpsk in zip(conn,kp):
            connsub=[]
            for ske in sk:
                sp = kpsk[ske[0]]
                ep = kpsk[ske[1]]
                if sp.sum()<=0 or ep.sum()<=0:
                        continue
                connsub.append(ske)
            connvis.append(connsub)

        imkpp_r = [draw_keypoints(im, kpim[None], colors="red", connectivity=sk,radius=8, width=8) for im,kpim,sk in zip(xy_rg,kpv,connvis)]

        ## kpts --> recon from gt
        kpor = data['test_kpts']
        kp = kpor.view(-1,*kpor.shape[2:])
        conn = data['train_skeleton'][0][0].long().tolist()
        conn= [conn]*kp.shape[0]
        vis= torch.ones((kp.shape[0],kp.shape[1],1)).to(scores.device)
        vis[kp.sum(axis=-1)==0]=0
        # conn=conn[kp.sum(axis=-1)==0]
        kpv = torch.cat((kp,vis),axis=-1)

        connvis= []
        for sk,kpsk in zip(conn,kp):
            connsub=[]
            for ske in sk:
                sp = kpsk[ske[0]]
                ep = kpsk[ske[1]]
                if sp.sum()<=0 or ep.sum()<=0:
                        continue
                connsub.append(ske)
            connvis.append(connsub)

        imkpp_g = [draw_keypoints(im, kpim[None], colors="green", connectivity=sk,radius=8, width=8) for im,kpim,sk in zip(xy_rg,kpv,connvis)]


        ims_train = []
        # bbgt --> recon from bbox
        train_images = data['train_images']
        train_bbox = data['train_bbox']
        train_kpts = data['train_kpts']
        for tim,tbox,kpor in zip(train_images,train_bbox,train_kpts):
            tim = tim[None]
            tbox = tbox[None]
            kpor= kpor[None]
            bbgt = tbox.view(-1,tbox.shape[-1])
            xx= unnorm(tim.view(-1,*tim.shape[2:])).to(torch.uint8)
            bbgt = tbox.view(-1,tbox.shape[-1])
            imtmp = [draw_bounding_boxes(im, torchvision.ops.box_convert(box[None], in_fmt='xywh', \
                                                                        out_fmt='xyxy'), colors='green', width=6) \
                                                                        for im,box in zip(xx,bbgt)]  
            kp = kpor.view(-1,*kpor.shape[2:])
            conn = data['train_skeleton'][0][0].long().tolist()
            conn= [conn]*kp.shape[0]
            vis= torch.ones((kp.shape[0],kp.shape[1],1)).to(scores.device)
            vis[kp.sum(axis=-1)==0]=0
            # conn=conn[kp.sum(axis=-1)==0]
            kpv = torch.cat((kp,vis),axis=-1)

            connvis= []
            for sk,kpsk in zip(conn,kp):
                connsub=[]
                for ske in sk:
                    sp = kpsk[ske[0]]
                    ep = kpsk[ske[1]]
                    if sp.sum()<=0 or ep.sum()<=0:
                            continue
                    connsub.append(ske)
                connvis.append(connsub)

            imtmp = [draw_keypoints(im, kpim[None], colors="green", connectivity=sk,radius=8, width=8) for im,kpim,sk in zip(imtmp,kpv,connvis)]
            ims_train.append(imtmp)

        
        xy_rg = torch.stack(xy_rg,axis=0)/255 *2 -1
        xy_b = torch.stack(xy_b,axis=0)/255 *2 -1
        kpg = torch.stack(imkpp_g,axis=0)/255 *2 -1
        kpr = torch.stack(imkpp_r,axis=0)/255 *2 -1
        kpb = torch.stack(imkpp_b,axis=0)/255 *2 -1

        imtrainvis = [torch.stack(x,axis=0)/255 *2 -1 for x in ims_train]

        # catlist = [xy_rg,xy_b,kpg,kpr,kpb]
        catlist = [imtrainvis[0],imtrainvis[1],kpg,kpb]
        kpim = torch.cat(catlist,axis=0)
        
        
        save_image(kpim, os.path.join(output_dir,name), nrow=4, normalize=True, range=(-1, 1))
        kpim = make_grid(kpim,nrow=4, normalize=True, range=(-1, 1))
        print('\n............Save a Sample.........\n')
        return kpim

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
