import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.trainers import BaseTrainer
import pdb
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes, draw_keypoints
import torchvision
# import src.utils.utilfunc as mut
import src.utils.utilfunc_awazi as mut
from data.tensordict import TensorDict
import glob
import imageio
from mmpose.core import imshow_bboxes, imshow_keypoints
import mmcv
from mmpose.datasets.dataset_info import DatasetInfo
import cv2
from data.tensordict import TensorDict
import time

class BoundingBoxDrawer:
    def __init__(self, im,vp):
        self.image = im
        self.image_copy = self.image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox_coordinates = []
        self.vp = vp
        cv2.namedWindow('Draw Bounding Boxes')
        cv2.setMouseCallback('Draw Bounding Boxes', self.draw_bbox)

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                cv2.rectangle(self.image_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                width = abs(x - self.ix)
                height = abs(y - self.iy)
                x_min = min(x, self.ix)
                y_min = min(y, self.iy)
                self.bbox_coordinates.append((x_min, y_min, width, height))
                self.drawing = False
        
    def run(self):
        while True:
            cv2.imshow('Draw Bounding Boxes', self.image_copy[:,:,::-1])
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save the bounding box coordinates to a file
                fname = self.vp.split('.')[0]+'bbox.npy'
                bbox = np.array(self.bbox_coordinates)
                np.save(fname,bbox)

            elif key == 27:  # Press ESC to exit
                break

        cv2.destroyAllWindows()

class KeypointsDrawer:
    def __init__(self, im,vp,cfg):
        self.image = im[:,:,::-1]
        self.image_copy = self.image.copy()
        self.vp = vp
        self.num_kps = cfg.data.settings.num_kpts
        self.keypoints = []
        self.window_name = 'Image with Keypoints'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        print('Hiii')
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:  # Check if the CTRL key is pressed
                self.keypoints.append((0,0))
            else:
                self.keypoints.append((x, y))
            if len(self.keypoints) == self.num_kps:
                self.draw_keypoints()
        

    def draw_keypoints(self):
        image_with_keypoints = np.copy(self.image)
        for i, kp in enumerate(self.keypoints):
            if isinstance(kp, tuple):  # Check if kp is a tuple (point)
                cv2.circle(image_with_keypoints, kp, 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint
                cv2.putText(image_with_keypoints, str(i), (kp[0] - 10, kp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:  
                cv2.putText(image_with_keypoints, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow(self.window_name, image_with_keypoints)
    
    def update_display(self):
        while True:
            self.draw_keypoints()
            key = cv2.waitKey(100)
            if len(self.keypoints) == self.num_kps:  # Press 'ESC' to exit
                self.keypoints =  np.array(self.keypoints).flatten()
                fname = self.vp.split('.')[0]+'kpts.npy'
                kpts = self.keypoints
                np.save(fname,kpts)
                break
        
    def run(self):
        cv2.imshow(self.window_name, self.image)
        self.update_display()
        cv2.destroyAllWindows()
       

    

def get_frames_from_video(video_path,cfg,pkpts=False,numframes=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()
    
    allframes= []
    start_idx=20
    curframe=0
    skip_frames = 1
    while True:
        ret, frame = cap.read()
        print('Frame: {}'.format(curframe),end='\r')
        if not ret:
            break
        height, width = frame.shape[:2]
        new_width = width
        new_height = height
        # new_width = int(width / 1.5)
        # new_height = int(height / 1.5)
        frame = cv2.resize(frame, (new_width, new_height))
        frame = frame[:,:,::-1]
        if curframe>start_idx and curframe%skip_frames==0:
            allframes.append(frame)
        # if len(allframes)>50:
        #     break
        curframe += 1
    
    print('Total frames in Video..{}\t Selecting {}'.format(len(allframes),numframes))
    allframes =allframes[skip_frames:]
    if numframes is not None:
        stride = len(allframes) // numframes
        allframes= allframes[::stride]
        
    fname = cfg.pipeline.video_path.split('.')[0]+'bbox.npy'
    if not os.path.exists(fname):
    # if True:
        im = allframes[0]
        inst = BoundingBoxDrawer(im,vp=cfg.pipeline.video_path)
        inst.run()
        
    bboxes = np.load(fname)
    
    if pkpts:
        fname = cfg.pipeline.video_path.split('.')[0]+'kpts.npy'
        if not os.path.exists(fname):
            im= allframes[0]
            inst = KeypointsDrawer(im,vp=cfg.pipeline.video_path,cfg=cfg)
            inst.run()
        kpts = np.load(fname)
        kpts = kpts[None]
        all_frames_dict= {
            'all_images': allframes,
            'all_bbox': bboxes,
            'all_kpts': kpts,
        }
    else:
        all_frames_dict= {
            'all_images': allframes,
            'all_bbox': bboxes,
        }
    return all_frames_dict
        

def get_frames(frames,images,cat,data_path,numkpts):
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
        # impath = data_path + os.sep.join(im.split(data_path.split(os.sep)[-1])[-1].split('\\'))
        impath = data_path + im
        bbox = np.array([x for x in data.bbox])
        kpts = np.array(data.keypoints).reshape(-1,3)[:,:2]
        kpts = kpts
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
        
    def generate_results_video(self,datain,pipeline,test_pipeline,ann_info,cfd,label_params,dataset_info,device,padding):
        torch.set_grad_enabled(self.training)
        all_im =[]
        num_train_frames = len(datain['train_images'])
        datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding)
        datas1={}
        data={}
        data['train_images'] = datatrain['train_images'][:,None]
        data['train_bbox'] = datatrain['train_bbox'][:,None]
        data['train_label'] = datatrain['train_label'][:,None]
        data['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
        data['train_skeleton'] = datatrain['train_skeleton'][:,None]
        initial_datatrain_copy = datatrain.copy()
        for idx,tim in enumerate(datain['test_images']):
            print('Processed... {}/{}'.format(idx,len(datain['test_images'])))
            # Process Stage-1
            datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding=3)
            # data = {}  # Resets added kpts labels, ltrb
            datas1['train_images'] = datatrain['train_images'][:,None]
            datas1['train_bbox'] = datatrain['train_bbox'][:,None]
            datas1['train_label'] = datatrain['train_label'][:,None]
            datas1['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
            datas1['train_skeleton'] = datatrain['train_skeleton'][:,None]
            
            datatest= mut.get_processed_testframe_giventraininfo([tim], datatrain ,test_pipeline,ann_info,cfd,label_params,dataset_info)
            datas1['test_images'] = datatest['test_images'][:,None]
            datas1 = TensorDict(datas1)
            datas1 = datas1.to(device)
            # t1 = time.time()
            target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=datas1['train_images'],
                                        test_imgs=datas1['test_images'],
                                        train_bb=datas1['train_bbox'],
                                        train_label=datas1['train_label'],
                                        train_kpts_label=datas1.get('train_kpts_label',None),
                                        train_ltrb_target=datas1['train_ltrb_target'],
                                        train_kpts_target= datas1.get('train_kpts_target',None),
                                        train_target = 'gmsp',
                                        )
            # t2 =time.time()
            # inf_time = t2 - t1
            # pdb.set_trace()
            
            kpim ,bbox,kpts, bboxorims1, kptsorim, _= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, datas1,imname='tmp.png',metas=datatrain['train_img_metas'][-1],image_orig=tim)
            
            # Process: Stage 2
            # data= {}
            datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding=2)
            data['train_images'] = datatrain['train_images'][:,None]
            data['train_bbox'] = datatrain['train_bbox'][:,None]
            data['train_label'] = datatrain['train_label'][:,None]
            data['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
            data['train_skeleton'] = datatrain['train_skeleton'][:,None]
            datatest,_= mut.get_processed_traininfo([tim], [bboxorims1[0]] ,pipeline,ann_info,cfd,label_params,dataset_info,padding=1.2,key_label='test_')
            data['test_images'] = datatest['test_images'][:,None]
            data = TensorDict(data)
            data = data.to(device)
            
            target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=data['train_images'],
                                        test_imgs=data['test_images'],
                                        train_bb=data['train_bbox'],
                                        train_label=data['train_label'],
                                        train_kpts_label=data.get('train_kpts_label',None),
                                        train_ltrb_target=data['train_ltrb_target'],
                                        train_kpts_target= data.get('train_kpts_target',None),
                                        train_target = 'gmsp',
                                        )
            kpim ,bbox,kpts, bboxorim, kptsorim, addframe= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, data,imname='tmp.png',metas=datatest['test_img_metas'][-1],image_orig=tim)
            # pdb.set_trace()
            # time.sleep(1)
            # kpim, bbox, kpts= self.visualize_target(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png')
            # all_im.append(np.uint8(kpim.permute(1,2,0).numpy()*255))
            all_im.append(kpim)

            # if idx<4: addframe=True
            addframe = True
            if addframe:
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
                    datain['train_images'] =            datain['train_images'][-num_train_frames:]
                    datain['train_bbox'] =              datain['train_bbox'][-num_train_frames:]
                    
                else:
                    datain['train_images'] = [datain['train_images'][0],datain['train_images'][-1]]
                    datain['train_bbox'] = [datain['train_bbox'][0],datain['train_bbox'][-1]]
                            
            else:
                # print('\nSKIPPED ADDING THIS FRAME\n')
                pass
                
                
        output_dir =  self.cfg.snaps.image_save_dir
        gifname = 'animal_{}.mp4'.format(0)
        gifname = os.path.join(output_dir,gifname)
        if type(all_im[0])==torch.Tensor:
            all_im = [(im.permute(1,2,0).numpy()*255).astype(np.uint8) for im in all_im]
        pdb.set_trace()
        imageio.mimsave(gifname,all_im,fps=24)
        # imageio.mimsave(gifname,all_im)
            

    def generate_results(self,data, animid=0,metas=None,vd=None,video_id=0,track_id=0):
        torch.set_grad_enabled(self.training)
        all_im =[]
        num_train_frames = data['train_images'].shape[0]
        for idx,tim in enumerate(data['test_images']):
            
            target_scores, bbox_preds, kp_preds, target_kpts_scores,_ = self.net(train_imgs=data['train_images'],
                                                test_imgs=tim[None,None],
                                                train_bb=data['train_bbox'],
                                                train_label=data['train_label'],
                                                train_kpts_label=data.get('train_kpts_label',None),
                                                train_ltrb_target=data['train_ltrb_target'],
                                                train_kpts_target= data.get('train_kpts_target',None),
                                                )
            

            data_tmp = TensorDict (
                        {
                        'train_images': data['train_images'],
                        'train_bbox': data['train_bbox'],
                        'train_skeleton': data['train_skeleton'],
                        'train_kpts': data.get('train_kpts',None),
                        'test_images': tim[None,None],
                        'test_label': data['test_label'][idx][None,None],
                        'test_ltrb_target' : data['test_ltrb_target'][idx][None,None],
                        'test_bbox' : data['test_bbox'][idx][None,None],
                        'test_kpts_target' : data['test_kpts_target'][idx][None,None] if data.get('test_kpts_target',None) is not None else None, # Verify.. not correct?
                        'test_kpts_label' : data['test_kpts_label'][idx][None,None] if data.get('test_kpts_label',None) is not None else None, # Verify.. not correct?
                        'test_kpts' : data['test_kpts'][idx][None,None] if data.get('test_kpts',None) is not None else None,
                        }
                    )   
            
            # kpim ,bbox,kpts= self.visualize_all_or(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png',metas=metas[idx])
            kpim, bbox, kpts= self.visualize_target(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png')
            # all_im.append(np.uint8(kpim.permute(1,2,0).numpy()*255))
            all_im.append(kpim)

            # data['train_images'][0] = data['train_images'][1]
            # data['train_bbox'][0] = data['train_bbox'][1]
            # data['train_kpts'][0] = data['train_kpts'][1] 

            if data.get('train_kpts',None) is not None:
                # data['train_bbox'][-1] = data['test_bbox'][idx][None,None]
                # data['train_kpts'][-1] = data['test_kpts'][idx][None,None]
                data['train_images'] = torch.cat((data['train_images'],tim[None,None]),dim=0)
                data['train_bbox']=torch.cat((data['train_bbox'], bbox[None,None]),dim=0)
                data['train_label']=torch.cat((data['train_label'],vd._generate_label_function(data['train_bbox'][-1])[None]),dim=0)
                tltrb, tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target']=torch.cat((data['train_ltrb_target'],tltrb[None]),dim=0)
                data['train_sample_region']=torch.cat((data['train_sample_region'],tsr[None]),dim=0)
        
                data['train_kpts']= torch.cat((data['train_kpts'],kpts[:,:,:2][None]),dim=0)
                data['train_kpts_label']=torch.cat((data['train_kpts_label'],vd._generate_kptlabel_function(data['train_kpts'][-1])[None]),dim=0)
                data['train_kpts_target']=torch.cat((data['train_kpts_target'], vd._generate_kpts_regression_targets(data['train_kpts'][-1])[None]),dim=0)
                
                # Slide
                kk=3
                data['train_images']= data['train_images'][-num_train_frames+kk:]
                data['train_bbox']= data['train_bbox'][-num_train_frames+kk:]
                data['train_label']= data['train_label'][-num_train_frames+kk:]
                data['train_ltrb_target']= data['train_ltrb_target'][-num_train_frames+kk:]
                data['train_sample_region']= data['train_sample_region'][-num_train_frames+kk:]
                data['train_kpts']= data['train_kpts'][-num_train_frames+kk:]
                data['train_kpts_label']= data['train_kpts_label'][-num_train_frames+kk:]
                data['train_kpts_target']= data['train_kpts_target'][-num_train_frames+kk:]
                
                # First and last
                # data['train_images']= data['train_images'][[0,-1]]
                # data['train_bbox']= data['train_bbox'][[0,-1]]
                # data['train_label']= data['train_label'][[0,-1]]
                # data['train_ltrb_target']= data['train_ltrb_target'][[0,-1]]
                # data['train_sample_region']= data['train_sample_region'][[0,-1]]
                # data['train_kpts']= data['train_kpts'][[0,-1]]
                # data['train_kpts_label']= data['train_kpts_label'][[0,-1]]
                # data['train_kpts_target']= data['train_kpts_target'][[0,-1]]
            else:
                data['train_images'] = tim.unsqueeze(0).unsqueeze(0).expand(num_train_frames,-1,-1,-1,-1)
                data['train_bbox'] = bbox.unsqueeze(0).unsqueeze(0).expand(num_train_frames,-1,-1)
                data['train_label'] = vd._generate_label_function(data['train_bbox'][-1]).expand(num_train_frames,-1,-1,-1)
                ltrbt,tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target'] = ltrbt.expand(num_train_frames,-1,-1,-1,-1)
                data['train_sample_region'] = tsr.expand(num_train_frames,-1,-1,-1,-1)
                
                data['train_kpts'] = kpts[:,:,:2][None].expand(num_train_frames,-1,-1,-1)
                data['train_kpts_label'] = vd._generate_kptlabel_function(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
                data['train_kpts_target']= vd._generate_kpts_regression_targets(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
            
            pdb.set_trace()    
                
        output_dir =  self.cfg.snaps.image_save_dir
        gifname = 'animal_{}.gif'.format(animid)
        gifname = os.path.join(output_dir,gifname)
        if type(all_im[0])==torch.Tensor:
            all_im = [(im.permute(1,2,0).numpy()*255).astype(np.uint8) for im in all_im]
        imageio.mimsave(gifname,all_im,duration=1000,loop=0)

    
    def visualize_all_eval(self,scores,bbop,kpop, target_kpts_scores, data, imname,metas,image_orig):
        ns = scores.shape[1]
        nf = scores.shape[0]
        # tlb = data['test_label']
        idsp = scores.reshape(nf, ns, -1).max(dim=2)[1]
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

        
        output_dir =  self.cfg.snaps.image_save_dir
        name = imname

        
        orim = image_orig
        center = metas['center']
        scale = metas['scale']
        # scale = scale * 200.0
        scale = scale * 200
        test_images = data['test_images']
        unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        img = xx[0].permute(1,2,0).cpu().numpy()
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
        print(score_pdkp.min())
        thresh= 0.60
        confidence = score_pdkp>thresh
        # if confidence.sum()> len(confidence)-15:
        if score_pdkp.min()>thresh:
            addframe=True
        else:
            print('Low Confidence Frame... {}'.format(confidence.sum()))
            addframe=False
        
        color_thresh =0.70
        if (score_pdkp>color_thresh).sum() > len(confidence)-6:
            color='green'
        else:
            color='blue'
        
        bpo=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors=color,thickness=2,show=False)
        # bpo_=mmcv.imshow_bboxes(img.copy(),bboxes.flatten()[None],colors='green',thickness=2,show=False)
        # plt.imsave(os.path.join(output_dir,name),bpo_)
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
        kpp = imshow_keypoints(img.copy(),kpv,skeleton=skeleton[0],
                              pose_kpt_color=pose_kpt_color[0],
                              pose_link_color=pose_link_color[0], 
                              radius=3,
                              thickness=2)
        # plt.imsave(os.path.join(output_dir,name), kpp)
        # pdb.set_trace()

        target_coords = np.ones_like(kpv[0])
        target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
        target_coords[:, 2] = kpv[0][:, 2] 
        connvis= []
        conncolorvis=[]
        for idx,(sk,kpsk,bb,plc) in enumerate(zip(conn,kp,bbpred,pose_link_color)):
            connsub=[]
            conncolor=[]
            for ske,plce in zip(sk,plc):
                sp = kpsk[ske[0]] 
                ep = kpsk[ske[1]] 
                if sp.sum()<=10:
                    continue
                if ep.sum()<=10:
                    continue
                bbox_x, bbox_y, bbox_w, bbox_h = bb.tolist()
                offset = 80
                bbox_x, bbox_y, bbox_w, bbox_h = (bbox_x - offset, bbox_y - offset, bbox_w + 2 * offset, bbox_h + 2 * offset)
                inside_ep = bbox_x <= ep[0].item() <= bbox_x + bbox_w and bbox_y <= ep[1].item() <= bbox_y + bbox_h
                inside_sp = bbox_x <= sp[0].item() <= bbox_x + bbox_w and bbox_y <= sp[1].item() <= bbox_y + bbox_h
                if inside_ep and inside_sp:
                    connsub.append(ske)
                    conncolor.append(plce)
            connvis.append(connsub)
            conncolorvis.append(conncolor)

        skel = np.array(connvis[0])
        plc= np.array(conncolorvis[0])
        kppo = imshow_keypoints(bpo,target_coords[None],skeleton=skel,
                              pose_kpt_color=pose_kpt_color[0],
                              pose_link_color=plc, 
                              radius=10,
                              thickness=5)
        
        ## kpts --> recon from gt
        kgto= None
        kpim = np.hstack((kgto,kppo)) if kgto is not None else kppo
        # print('\n............Save a Sample.........\n')
        # plt.imsave(os.path.join(output_dir,name), kpim)
        return kpim, bboxes_pred, kpv_pred, bboxes_orim_xyxy[None], target_coords[None], addframe
    
    
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
        bgt=mmcv.imshow_bboxes(img,bboxes,colors='green',thickness=2,show=False)
        
        orim = plt.imread(metas['image_file'])
        center = metas['center']
        scale = metas['scale']
        # scale = scale * 200.0
        scale = scale * 200
        output_size = img.shape
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
        bboxes = bboxes.reshape(-1,2)
        
        target_coords = np.ones_like(bboxes)
        target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        bboxes_orim = target_coords.flatten()[None]
        bgto=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors='green',thickness=2,show=False)

        # bbpred --> recon from ltrb
        bbpred = predbb_xywh.view(-1,predbb_xywh.shape[-1])
        bboxes_pred = bbpred[0]
        bboxes = torchvision.ops.box_convert(bbpred[0][None], in_fmt='xywh', out_fmt='xyxy').cpu().numpy().astype(int)
        bboxes = bboxes.reshape(-1,2)
        target_coords = np.ones_like(bboxes)
        target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        bboxes_orim = target_coords.flatten()[None]
        bpo=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors='green',thickness=2,show=False)

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
        print(score_pdkp.min())
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
        kpp = imshow_keypoints(img.copy(),kpv,skeleton=skeleton[0],
                              pose_kpt_color=pose_kpt_color[0],
                              pose_link_color=pose_link_color[0], 
                              radius=3,
                              thickness=2)

        target_coords = np.ones_like(kpv[0])
        target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
        target_coords[:, 2] = kpv[0][:, 2] 
        connvis= []
        conncolorvis=[]
        for idx,(sk,kpsk,bb,plc) in enumerate(zip(conn,kp,bbpred,pose_link_color)):
            connsub=[]
            conncolor=[]
            for ske,plce in zip(sk,plc):
                sp = kpsk[ske[0]] 
                ep = kpsk[ske[1]] 
                if sp.sum()<=10:
                    continue
                if ep.sum()<=10:
                    continue
                bbox_x, bbox_y, bbox_w, bbox_h = bb.tolist()
                offset = 5
                bbox_x, bbox_y, bbox_w, bbox_h = (bbox_x - offset, bbox_y - offset, bbox_w + 2 * offset, bbox_h + 2 * offset)

                inside_ep = bbox_x <= ep[0].item() <= bbox_x + bbox_w and bbox_y <= ep[1].item() <= bbox_y + bbox_h
                inside_sp = bbox_x <= sp[0].item() <= bbox_x + bbox_w and bbox_y <= sp[1].item() <= bbox_y + bbox_h
                if inside_ep and inside_sp:
                    connsub.append(ske)
                    conncolor.append(plce)
            connvis.append(connsub)
            conncolorvis.append(conncolor)

        skel = np.array(connvis[0])
        plc= np.array(conncolorvis[0])
        kppo = imshow_keypoints(bpo,target_coords[None],skeleton=skel,
                              pose_kpt_color=pose_kpt_color[0],
                              pose_link_color=plc, 
                              radius=10,
                              thickness=5)
        
        ## kpts --> recon from gt
        kpor = data['test_kpts']
        kgto= None
        if kpor is not None:
            kp = kpor.view(-1,*kpor.shape[2:])
            conn = data['train_skeleton'][0][0].long().tolist()
            conn= [conn]*kp.shape[0]
            vis= torch.ones((kp.shape[0],kp.shape[1],1)).to(scores.device)
            vis[kp.sum(axis=-1)==0]=0
            # conn=conn[kp.sum(axis=-1)==0]
            kpv = torch.cat((kp,vis),axis=-1)
            kpv= kpv.cpu().numpy()
            kgt = imshow_keypoints(img.copy(),kpv,skeleton=skeleton[0],
                                pose_kpt_color=pose_kpt_color[0],
                                pose_link_color=pose_link_color[0], 
                                radius=3,
                                thickness=2)

            target_coords = np.ones_like(kpv[0])
            target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
            target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
            target_coords[:, 2] = kpv[0][:, 2] 

            kgto = imshow_keypoints(bgto,target_coords[None],skeleton=skeleton[0],
                                pose_kpt_color=pose_kpt_color[0],
                                pose_link_color=pose_link_color[0], 
                                radius=10,
                                thickness=5)
        
        kpim = np.hstack((kgto,kppo)) if kgto is not None else kppo
        print('\n............Save a Sample.........\n')
        plt.imsave(os.path.join(output_dir,name), kpim)
        return kpim, bboxes_pred, kpv_pred #, bboxes_gt
    
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
        bboxes_pred = bbpred[0]
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
        kpv_pred= kpv
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
        if kpgtop is not None:
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
        if kpor is not None:
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
        if train_kpts is not None:
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
        if data['test_kpts'] is not None:
            kpg = torch.stack(imkpp_g,axis=0)/255 *2 -1
        if kpgtop is not None:
            kpr = torch.stack(imkpp_r,axis=0)/255 *2 -1
        kpb = torch.stack(imkpp_b,axis=0)/255 *2 -1

        imtrainvis = [torch.stack(x,axis=0)/255 *2 -1 for x in ims_train]

        # catlist = [xy_rg,xy_b,kpg,kpr,kpb]
        if kpgtop is not None:
            catlist = [imtrainvis[0],imtrainvis[1],kpg,kpb]
        else:
            if len(imtrainvis)>1:
                catlist = [imtrainvis[0],imtrainvis[1],kpb]
            else:
                catlist = [kpb,kpb,kpb]
        kpim = torch.cat(catlist,axis=0)
        
        
        save_image(kpim, os.path.join(output_dir,name), nrow=4, normalize=True, range=(-1, 1))
        kpim = make_grid(kpim,nrow=4, normalize=True, range=(-1, 1))
        print('\n............Save a Sample.........\n')
        return kpim, bboxes_pred, kpv_pred

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
