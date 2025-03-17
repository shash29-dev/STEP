import os
import time
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
import cv2
from data.tensordict import TensorDict


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
        # elif event == cv2.EVENT_LBUTTONUP:
        #     if self.drawing:
        #         cv2.rectangle(self.image_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        #         self.bbox_coordinates.append((min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y)))
        #         self.drawing = False

    def run(self):
        while True:
            cv2.imshow('Draw Bounding Boxes', self.image_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save the bounding box coordinates to a file
                fname = self.vp.split('.')[0]+'.npy'
                bbox = np.array(self.bbox_coordinates)
                np.save(fname,bbox)

            elif key == 27:  # Press ESC to exit
                break

        cv2.destroyAllWindows()


def get_frames_from_video(video_path,cfg,pkpts=False,numframes=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()
    
    allframes= []
    start_idx=0
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
        if len(allframes)>50:
            break
        curframe += 1
    
    print('Total frames in Video..{}\t Selecting {}'.format(len(allframes),numframes))
    allframes =allframes[skip_frames:]
    if numframes is not None:
        stride = len(allframes) // numframes
        allframes= allframes[::stride]
        
    fname = cfg.pipeline.video_path.split('.')[0]+'bbox.npy'
    if not os.path.exists(fname):
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
        

def get_frames(frames,images,cat,data_path):
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
        impath = data_path + os.sep.join(im.split(data_path.split(os.sep)[-1])[-1].split('\\'))
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
    
    def run_tracking(self,datain,pipeline,test_pipeline,ann_info,cfd,label_params,dataset_info,device):
        datatrain,vd = mut.get_processed_traininfofull(datain['train_images'],datain['train_bbox'],test_pipeline,ann_info,cfd,label_params,dataset_info)
        data={}
        all_im=[]
        num_train_frames = len(datain['train_images'])
        data['train_images'] = datatrain['train_images'][:,None]
        data['train_bbox'] = datatrain['train_bbox'][:,None]
        data['train_label'] = datatrain['train_label'][:,None]
        data['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
        data['train_skeleton'] = datatrain['train_skeleton'][:,None]
        for idx,tim in enumerate(datain['test_images']):
            print('Processed... {}/{}'.format(idx,len(datain['test_images'])))
            datatest= mut.get_processed_testframe_full([tim] ,test_pipeline,ann_info,cfd,label_params,dataset_info)
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
                                                )
            data_tmp = TensorDict (
                        {
                        'train_images': data['train_images'],
                        'train_bbox': data['train_bbox'],
                        'train_skeleton': data['train_skeleton'],
                        'train_kpts': data.get('train_kpts',None),
                        'test_images': data['test_images'],
                        'test_label': None,
                        'test_ltrb_target' : None,
                        'test_bbox' : None,
                        'test_kpts_target' : data['test_kpts_target'][idx][None,None] if data.get('test_kpts_target',None) is not None else None, # Verify.. not correct?
                        'test_kpts_label' : data['test_kpts_label'][idx][None,None] if data.get('test_kpts_label',None) is not None else None, # Verify.. not correct?
                        'test_kpts' : data['test_kpts'][idx][None,None] if data.get('test_kpts',None) is not None else None,
                        }
                    )   
            kpim ,bbox,kpts, bboxorim, kptsorim, addframe= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png',metas=datatest['test_img_metas'][0],image_orig=tim)
            all_im.append(kpim)

            if idx<10: addframe=True
            
            if data.get('train_kpts',None) is not None and addframe:
                # data['train_bbox'][-1] = data['test_bbox'][idx][None,None]
                # data['train_kpts'][-1] = data['test_kpts'][idx][None,None]
                datain['train_images'].append(tim)
                datain['train_bbox'].append(bboxorim[0])
                
                data['train_images'] = torch.cat((data['train_images'],data['test_images']),dim=0)
                data['train_bbox']=torch.cat((data['train_bbox'], bbox[None,None]),dim=0)
                data['train_label']=torch.cat((data['train_label'],vd._generate_label_function(data['train_bbox'][-1])[None]),dim=0)
                tltrb, tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target']=torch.cat((data['train_ltrb_target'],tltrb[None]),dim=0)
                data['train_sample_region']=torch.cat((data['train_sample_region'],tsr[None]),dim=0)
        
                data['train_kpts']= torch.cat((data['train_kpts'],kpts[:,:,:2][None]),dim=0)
                data['train_kpts_label']=torch.cat((data['train_kpts_label'],vd._generate_kptlabel_function(data['train_kpts'][-1])[None]),dim=0)
                data['train_kpts_target']=torch.cat((data['train_kpts_target'], vd._generate_kpts_regression_targets(data['train_kpts'][-1])[None]),dim=0)
                
                # Slide
                # datain['train_images'] =            datain['train_images'][-num_train_frames:]
                # datain['train_bbox'] =              datain['train_bbox'][-num_train_frames:]
                # data['train_images']=               data['train_images'][-num_train_frames:]
                # data['train_bbox']=                 data['train_bbox'][-num_train_frames:]
                # data['train_label']=                data['train_label'][-num_train_frames:]
                # data['train_ltrb_target']=          data['train_ltrb_target'][-num_train_frames:]
                # data['train_sample_region']=        data['train_sample_region'][-num_train_frames:]
                # data['train_kpts']=                 data['train_kpts'][-num_train_frames:]
                # data['train_kpts_label']=           data['train_kpts_label'][-num_train_frames:]
                # data['train_kpts_target']=          data['train_kpts_target'][-num_train_frames:]
                
                # First and last
                datain['train_images'] = [datain['train_images'][0],datain['train_images'][-1]]
                datain['train_bbo'] = [datain['train_bbox'][0],datain['train_bbox'][-1]]
                data['train_images']= data['train_images'][[0,-1]]
                data['train_bbox']= data['train_bbox'][[0,-1]]
                data['train_label']= data['train_label'][[0,-1]]
                data['train_ltrb_target']= data['train_ltrb_target'][[0,-1]]
                data['train_sample_region']= data['train_sample_region'][[0,-1]]
                data['train_kpts']= data['train_kpts'][[0,-1]]
                data['train_kpts_label']= data['train_kpts_label'][[0,-1]]
                data['train_kpts_target']= data['train_kpts_target'][[0,-1]]
            elif data.get('train_kpts',None) is  None:
                print('\n\nRepaeting... No sliding frames...\n\n')
                data['train_images'] = data['test_images'].expand(num_train_frames,-1,-1,-1,-1)
                data['train_bbox'] = bbox.unsqueeze(0).unsqueeze(0).expand(num_train_frames,-1,-1)
                data['train_label'] = vd._generate_label_function(data['train_bbox'][-1]).expand(num_train_frames,-1,-1,-1)
                ltrbt,tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target'] = ltrbt.expand(num_train_frames,-1,-1,-1,-1)
                data['train_sample_region'] = tsr.expand(num_train_frames,-1,-1,-1,-1)
                
                data['train_kpts'] = kpts[:,:,:2][None].expand(num_train_frames,-1,-1,-1)
                data['train_kpts_label'] = vd._generate_kptlabel_function(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
                data['train_kpts_target']= vd._generate_kpts_regression_targets(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
            else:
                print('\n\n SKIPPED ADDING THIS FRAME\n\n')
                pass
            
                
        output_dir =  self.cfg.snaps.image_save_dir
        gifname = 'animal_{}.mp4'.format(0)
        gifname = os.path.join(output_dir,gifname)
        if type(all_im[0])==torch.Tensor:
            all_im = [(im.permute(1,2,0).numpy()*255).astype(np.uint8) for im in all_im]
        imageio.mimsave(gifname,all_im,fps=5)
        pdb.set_trace()
    
    def generate_results_video(self,datain,pipeline,test_pipeline,ann_info,cfd,label_params,dataset_info,device,padding):
        torch.set_grad_enabled(self.training)
        all_im =[]
        num_train_frames = len(datain['train_images'])
        get_tracking_bbox = self.run_tracking(datain,pipeline,test_pipeline,ann_info,cfd,label_params,dataset_info,device)
        datatrain,vd = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding)
        data={}
        data['train_images'] = datatrain['train_images'][:,None]
        data['train_bbox'] = datatrain['train_bbox'][:,None]
        data['train_label'] = datatrain['train_label'][:,None]
        data['train_ltrb_target'] = datatrain['train_ltrb_target'][:,None]
        data['train_skeleton'] = datatrain['train_skeleton'][:,None]
        initial_datatrain_copy = datatrain.copy()
        for idx,tim in enumerate(datain['test_images']):
            print('Processed... {}/{}'.format(idx,len(datain['test_images'])))
            datatest= mut.get_processed_testframe_giventraininfo([tim], datatrain ,test_pipeline,ann_info,cfd,label_params,dataset_info)
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
                                                )
            data_tmp = TensorDict (
                        {
                        'train_images': data['train_images'],
                        'train_bbox': data['train_bbox'],
                        'train_skeleton': data['train_skeleton'],
                        'train_kpts': data.get('train_kpts',None),
                        'test_images': data['test_images'],
                        'test_label': None,
                        'test_ltrb_target' : None,
                        'test_bbox' : None,
                        'test_kpts_target' : data['test_kpts_target'][idx][None,None] if data.get('test_kpts_target',None) is not None else None, # Verify.. not correct?
                        'test_kpts_label' : data['test_kpts_label'][idx][None,None] if data.get('test_kpts_label',None) is not None else None, # Verify.. not correct?
                        'test_kpts' : data['test_kpts'][idx][None,None] if data.get('test_kpts',None) is not None else None,
                        }
                    )   
            
            kpim ,bbox,kpts, bboxorim, kptsorim, addframe= self.visualize_all_eval(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png',metas=datatrain['train_img_metas'][0],image_orig=tim)
            # pdb.set_trace()
            time.sleep(1)
            # kpim, bbox, kpts= self.visualize_target(target_scores,bbox_preds, kp_preds, target_kpts_scores, data_tmp,imname='tmp.png')
            # all_im.append(np.uint8(kpim.permute(1,2,0).numpy()*255))
            all_im.append(kpim)

            # data['train_images'][0] = data['train_images'][1]
            # data['train_bbox'][0] = data['train_bbox'][1]
            # data['train_kpts'][0] = data['train_kpts'][1] 

            if idx<10: addframe=True
            
            if data.get('train_kpts',None) is not None and addframe:
                # data['train_bbox'][-1] = data['test_bbox'][idx][None,None]
                # data['train_kpts'][-1] = data['test_kpts'][idx][None,None]
                datain['train_images'].append(tim)
                datain['train_bbox'].append(bboxorim[0])
                
                data['train_images'] = torch.cat((data['train_images'],data['test_images']),dim=0)
                data['train_bbox']=torch.cat((data['train_bbox'], bbox[None,None]),dim=0)
                data['train_label']=torch.cat((data['train_label'],vd._generate_label_function(data['train_bbox'][-1])[None]),dim=0)
                tltrb, tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target']=torch.cat((data['train_ltrb_target'],tltrb[None]),dim=0)
                data['train_sample_region']=torch.cat((data['train_sample_region'],tsr[None]),dim=0)
        
                data['train_kpts']= torch.cat((data['train_kpts'],kpts[:,:,:2][None]),dim=0)
                data['train_kpts_label']=torch.cat((data['train_kpts_label'],vd._generate_kptlabel_function(data['train_kpts'][-1])[None]),dim=0)
                data['train_kpts_target']=torch.cat((data['train_kpts_target'], vd._generate_kpts_regression_targets(data['train_kpts'][-1])[None]),dim=0)
                
                # Slide
                datain['train_images'] =            datain['train_images'][-num_train_frames:]
                datain['train_bbox'] =              datain['train_bbox'][-num_train_frames:]
                data['train_images']=               data['train_images'][-num_train_frames:]
                data['train_bbox']=                 data['train_bbox'][-num_train_frames:]
                data['train_label']=                data['train_label'][-num_train_frames:]
                data['train_ltrb_target']=          data['train_ltrb_target'][-num_train_frames:]
                data['train_sample_region']=        data['train_sample_region'][-num_train_frames:]
                data['train_kpts']=                 data['train_kpts'][-num_train_frames:]
                data['train_kpts_label']=           data['train_kpts_label'][-num_train_frames:]
                data['train_kpts_target']=          data['train_kpts_target'][-num_train_frames:]
                
                # First and last
                # datain['train_images'] = [datain['train_images'][0],datain['train_images'][-1]]
                # datain['train_bbo'] = [datain['train_bbox'][0],datain['train_bbox'][-1]]
                # data['train_images']= data['train_images'][[0,-1]]
                # data['train_bbox']= data['train_bbox'][[0,-1]]
                # data['train_label']= data['train_label'][[0,-1]]
                # data['train_ltrb_target']= data['train_ltrb_target'][[0,-1]]
                # data['train_sample_region']= data['train_sample_region'][[0,-1]]
                # data['train_kpts']= data['train_kpts'][[0,-1]]
                # data['train_kpts_label']= data['train_kpts_label'][[0,-1]]
                # data['train_kpts_target']= data['train_kpts_target'][[0,-1]]
            elif data.get('train_kpts',None) is  None:
                print('\n\nRepaeting... No sliding frames...\n\n')
                data['train_images'] = data['test_images'].expand(num_train_frames,-1,-1,-1,-1)
                data['train_bbox'] = bbox.unsqueeze(0).unsqueeze(0).expand(num_train_frames,-1,-1)
                data['train_label'] = vd._generate_label_function(data['train_bbox'][-1]).expand(num_train_frames,-1,-1,-1)
                ltrbt,tsr = vd._generate_ltbr_regression_targets(data['train_bbox'][-1])
                data['train_ltrb_target'] = ltrbt.expand(num_train_frames,-1,-1,-1,-1)
                data['train_sample_region'] = tsr.expand(num_train_frames,-1,-1,-1,-1)
                
                data['train_kpts'] = kpts[:,:,:2][None].expand(num_train_frames,-1,-1,-1)
                data['train_kpts_label'] = vd._generate_kptlabel_function(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
                data['train_kpts_target']= vd._generate_kpts_regression_targets(data['train_kpts'][-1]).expand(num_train_frames,-1,-1,-1,-1)
            else:
                print('\n\n SKIPPED ADDING THIS FRAME\n\n')
                pass
            
            # Updates cropping center? --> Check
            if data.get('train_kpts',None) is not None and addframe:
                datatrain,_ = mut.get_processed_traininfo(datain['train_images'],datain['train_bbox'],pipeline,ann_info,cfd,label_params,dataset_info,padding)
            # datain['train_images'] = data['train_images']
            # datain['train_bbox'] = data['train_bbox']
                
                
        output_dir =  self.cfg.snaps.image_save_dir
        gifname = 'animal_{}.gif'.format(0)
        gifname = os.path.join(output_dir,gifname)
        if type(all_im[0])==torch.Tensor:
            all_im = [(im.permute(1,2,0).numpy()*255).astype(np.uint8) for im in all_im]
        imageio.mimsave(gifname,all_im[30:],duration=1000* 1/10,loop=0)
    
        
    def visualize_all_eval(self,scores,bbop,kpop, target_kpts_scores, data, imname,metas,image_orig):
        ns = scores.shape[1]
        nf = scores.shape[0]
        tlb = data['test_label']
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
        center = metas.get('center',None)
        scale = metas.get('scale',None)
        # scale = scale * 200.0
        test_images = data['test_images']
        unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        img = xx[0].permute(1,2,0).cpu().numpy()
        output_size = data['train_images'].shape[-2:]
        if scale is not None:
            scale = scale * 200
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]
            
        # bbpred --> recon from ltrb
        bbpred = predbb_xywh.view(-1,predbb_xywh.shape[-1])
        bboxes_pred = bbpred[0]
        bboxes = torchvision.ops.box_convert(bbpred[0][None], in_fmt='xywh', out_fmt='xyxy').cpu().numpy().astype(int)
        bboxes = bboxes.reshape(-1,2)
        
        # bboxes =  data['train_bbox'][0][0].cpu().numpy().reshape(-1,2).astype(int)
        
        if metas.get('M_inv',None) is  None:
            target_coords = np.ones_like(bboxes)
            target_coords[:, 0] = bboxes[:, 0] * scale_x + center[0] - scale[0] * 0.5
            target_coords[:, 1] = bboxes[:, 1] * scale_y + center[1] - scale[1] * 0.5
        else:
            target_coords = mut.transform_bbox_affine_matrix(bboxes.flatten().tolist(), metas['M_inv'])
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
        thresh= 0.30
        confidence = score_pdkp>thresh
        # if confidence.sum()> len(confidence)-15:
        if score_pdkp.min()>thresh:
            addframe=True
        else:
            print('Low Confidence Frame... {}'.format(confidence.sum()))
            addframe=False
        
        color_thresh =0.50
        if (score_pdkp>0.40).sum() > len(confidence)-6:
            color='green'
        else:
            color='blue'
        
        bpo=mmcv.imshow_bboxes(orim.copy(),bboxes_orim,colors=color,thickness=2,show=False)
        bpo_=mmcv.imshow_bboxes(img.copy(),bboxes.flatten()[None],colors='green',thickness=2,show=False)
        plt.imsave(os.path.join(output_dir,name),bpo)
        # plt.imsave(os.path.join(output_dir,name),bpo)
        
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
        if metas.get('M_inv',None) is  None:
            target_coords[:, 0] = kpv[0][:, 0] * scale_x + center[0] - scale[0] * 0.5
            target_coords[:, 1] = kpv[0][:, 1] * scale_y + center[1] - scale[1] * 0.5
        else:
            target_coords = kpv[0]
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
        kgto= None
        kpim = np.hstack((kgto,kppo)) if kgto is not None else kppo
        print('\n............Save a Sample.........\n')
        plt.imsave(os.path.join(output_dir,name), kpim)
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

            

