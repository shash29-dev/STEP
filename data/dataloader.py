# -*- coding: utf-8 -*-
import copy
import importlib
import itertools
import pickle
import random
import numpy as np
import pdb
import torch
import os
from data.tensordict import TensorDict
import pandas as pd
import torchvision
from mmpose.datasets.dataset_info import DatasetInfo
from xtcocotools.coco import COCO
from data.mmpipeline_trans import train_pipeline, val_pipeline
from mmpose.datasets.pipelines import Compose as mpCompose
import os.path as osp
import src.utils.utilfunc as mut
import data.processing_utils as prutils

class APT36Kmm(torch.utils.data.Dataset):
    def __init__(self, cfd, samples_per_videos,
                 num_test_frames, num_train_frames=1,
                 label_function_params=None,
                 stride=16,
                 coco_style=True):
        kw = cfd.kwargs
        self.esf=0

        self.cfd= cfd
        self.output_sz = cfd.settings.output_sz
        self.stride= stride
        self.use_normalized_coords = cfd.settings.normalized_bbreg_coords
        self.center_sampling_radius = cfd.settings.center_sampling_radius
        self.test_mode = ~cfd.settings.train
        self.img_prefix = kw.data_path
        self.ann_info = {}
        annotjson = cfd.settings.annotjson
        self.ann_file = annotjson
        self.get_nearinterval_frames = cfd.settings.get('near_interval_frame', False)
        self.ann_info['image_size'] = np.array([cfd.settings.output_sz,cfd.settings.output_sz])
        # self.ann_info['image_size'] = np.array([192,256])
        self.ann_info['heatmap_size'] = self.ann_info['image_size']//4
        self.ann_info['use_different_joint_weights'] = cfd.settings.get('use_different_joint_weights', False)
        self.ann_info['num_joints'] = cfd.settings.num_kpts
        self.ann_info['num_output_channels'] = cfd.settings.num_kpts
        dataset_info_path = cfd.settings.dataset_info
        dcname = dataset_info_path.split(os.sep)[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(dcname, dataset_info_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset_info = getattr(module,dcname)

        dataset_info = DatasetInfo(dataset_info)
        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.num_all_frames = num_train_frames + num_test_frames
        self.label_function_params=label_function_params
        if coco_style:
            self.coco = COCO(self.ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict((self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:])
            
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id, self.samevidframes = self._get_mapping_id_name(self.coco.imgs, cfd ,self.ann_file, clean_data_path=cfd.settings.clean_data_path)
        self.db = []
        if cfd.settings.train==True:
            pipeline= train_pipeline
        else:
            pipeline= val_pipeline
        
        self.pipeline = mpCompose(pipeline)
        self.db, self.id2Cat = self._get_db()


    def _get_db(self):
        gt_db, id2Cat = self._load_coco_keypoint_annotations()
        return gt_db, id2Cat
    
    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db, id2Cat = [], dict()
        ## Edit for video sequence here
        # for idx,img_id in enumerate(self.img_ids[:100]):
        for idx,img_id in enumerate(self.img_ids):
            print('Making Database... {}/{}'.format(idx,len(self.img_ids)))
            db_tmp, id2Cat_tmp = self._load_coco_keypoint_annotation_kernel(img_id)
            gt_db.extend(db_tmp)
            id2Cat.update({img_id: id2Cat_tmp})
            
        return gt_db, id2Cat
    
    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        objs = self.coco.loadAnns(ann_ids)

        objs = self.sanitize_boxes(objs,height,width)

        rec = []
        id2Cat = []
        if 'crowdpose' in self.cfd.kind: 
            allkps = [obj['keypoints'] for obj in objs]
            allkps = [item for sublist in allkps for item in sublist]
            for obj in objs:
                obj['all_keypoints'] = allkps

        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] < self.cfd.settings.min_kps:
                continue
            idnf_rec_tmp= []
            idnf2Cat_tmp= []
            bbox_id = obj['track_id']
            # rec_e, id2cat_e = self.get_info(obj,img_id)
            # idnf_rec_tmp.append(rec_e)
            # idnf2Cat_tmp.append(id2cat_e)
            svtobjs = self.getsameVidFrame(imgIds=img_id,bboxId=bbox_id,iscrowd=None)
            if 'aptmmpose' in self.cfd.kind or 'apt10k' in self.cfd.kind or 'mousepose' in self.cfd.kind: 
                if len(svtobjs)>=self.num_all_frames:
                    svf = random.sample(svtobjs, self.num_all_frames)
                else: 
                    self.esf +=1
                    if len(svtobjs)==0:
                        objnew= obj.copy()
                        objnew['all_keypoints']=objnew['keypoints']
                        print('####')
                        svtobjs.extend([objnew]*(self.num_all_frames - len(svtobjs)))
                    else:
                        svtobjs.extend([svtobjs[-1]]*(self.num_all_frames - len(svtobjs)))
                        print('*****')
                    svf =svtobjs
            elif 'crowdpose' in self.cfd.kind:
                svtobjs.extend([obj]*(self.num_all_frames - len(svtobjs)))
                svf =svtobjs
            else:
                raise('Implement...')

            # svfobjs = self.coco.loadAnns(svf)
            svfobjs = self.sanitize_boxes(svf,height,width)  # What if bbox is deleted after sanitizing? Append until len(rec)==self.num_all_frames
            for svfobj in svfobjs:
                svfid = svfobj['image_id']
                rec_svfe, id2cat_svfe = self.get_info(svfobj,svfid)
                idnf_rec_tmp.append(rec_svfe)
                idnf2Cat_tmp.append(id2cat_svfe)
           
            if len(idnf2Cat_tmp)<self.num_all_frames:
                print('After Sanitization, box deleted.. appending')
                while len(idnf2Cat_tmp)< self.num_all_frames: idnf2Cat_tmp.append(idnf2Cat_tmp[-1])
                while len(idnf_rec_tmp)< self.num_all_frames: idnf_rec_tmp.append(idnf_rec_tmp[-1])
            rec.append(idnf_rec_tmp)
            id2Cat.append(idnf2Cat_tmp)
            # print([x['track_id'] for x in svfobjs])
        
        for ele in rec:
            try:
                assert len(ele)==self.num_all_frames
            except:
                pdb.set_trace()
        return rec, id2Cat

    def get_info(self,obj,img_id):
        bbox_id = obj['track_id']
        num_joints = self.ann_info['num_joints']
        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
        joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

        keypoints = np.array(obj['keypoints']).reshape(-1, 3)
        nkps = len(obj['all_keypoints'])//(keypoints.shape[0]*keypoints.shape[1])
        gmsp_kps = np.array(obj['all_keypoints']).reshape(nkps,-1,3)
        gmsp_kps_joints = np.zeros(gmsp_kps.shape, dtype=np.float32)
        gmsp_kps_joints_visible = np.zeros(gmsp_kps.shape, dtype=np.float32)

        if  'fishpose' in self.cfd.kind:
            keypoints= keypoints[:self.ann_info['num_joints']]
            
        joints_3d[:, :2] = keypoints[:, :2]
        joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

        gmsp_kps_joints[:,:,:2] = gmsp_kps[:,:,:2]
        for idxgmsp, tmpgmspkp in enumerate(gmsp_kps):
            gmsp_kps_joints_visible[idxgmsp,:, :2] = np.minimum(1, tmpgmspkp[:, 2:3])

        # Convert to ltrb here?
        center, scale = self._xywh2cs(*obj['clean_bbox'][:4])
        image_file = osp.join(self.img_prefix, self.id2name[img_id])

        rec = {
            'image_file': image_file,
            'center': center,
            'scale': scale,
            'bbox': obj['clean_bbox'][:4],
            'rotation': 0,
            'joints_3d': joints_3d,
            'joints_3d_visible': joints_3d_visible,
            'dataset': self.dataset_name,
            'bbox_score': 1,
            'bbox_id': bbox_id,
            'gmsp_kps_joints': gmsp_kps_joints,
            'gmsp_kps_joints_visible': gmsp_kps_joints_visible,
        }
        category = obj['category_id']
        id2Cat = {
            'image_file': image_file,
            'bbox_id': bbox_id,
            'category': category,
        }
        return rec,id2Cat

    def getsameVidFrame(self,imgIds,bboxId,iscrowd):
        videoframes = self.samevidframes[imgIds]
        if self.get_nearinterval_frames:
            videoframes = np.array(videoframes)
            videoframes = videoframes[np.abs(videoframes-imgIds)<=self.get_nearinterval_frames].tolist()

        videoframes = list(itertools.chain.from_iterable([self.coco.getAnnIds(vidf) for vidf in videoframes]))
        lists = [self.coco.loadAnns(vidf) for vidf in videoframes if vidf in self.coco.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))
        anns= pd.DataFrame(anns)
        if anns.empty:
            return []

        all_kps = pd.DataFrame(anns.groupby('image_id')['keypoints'].apply(lambda x: [item for sublist in x for item in sublist]))
        all_kps.rename(columns={'keypoints': 'all_keypoints'}, inplace=True)
        anns = pd.merge(anns, all_kps, on='image_id', how='inner')
        anns_tid = anns[anns['track_id']==bboxId]

        
        anns_tid = anns_tid[anns_tid['num_keypoints']> self.cfd.settings.min_kps]
        
        anns_tid =  anns_tid.to_dict(orient='records')

        return anns_tid

    def sanitize_boxes(self,objs,height,width):
        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        return valid_objs
    
        
    def __len__(self):
        return len(self.db)
    
    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info['image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        offset = min(self.ann_info['image_size'][0],self.ann_info['image_size'][1])//self.cfd.settings.offset_factor
        ox,oy =  np.random.randint(low=-offset,high=offset),  np.random.randint(low=-offset,high=offset)
        center[0] = center[0]+ox
        center[1] = center[1]+oy
        # print(ox,oy)
        if (not self.test_mode) and np.random.rand() < 0.3: center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        # parsc = np.random.randint(low=100,high=200)
        parsc = np.random.randint(low=200,high=340)
        if 'crowdpose' in self.cfd.kind: 
            parsc = np.random.randint(low=300,high=600)
        scale = np.array([w / parsc, h / parsc], dtype=np.float32)
        # scale = np.array([1,1], dtype=np.float32)
        # padding to include proper amount of context
        padding = np.random.uniform(1.5,4)
        # if self.cfd.kind != 'jrdbpose':
        #     raise NotImplementedError('Correct padding')
        scale = scale * padding
        return center, scale
    
    @staticmethod
    def _get_mapping_id_name(imgs, cfd,aif=None, clean_data_path = False):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        samevidframes= {}
        computesvf = True
        if os.path.exists('datacache/'+cfd.kind+'/samevidfr_apt36k.pkl'):
            computesvf=False
            with open('datacache/'+cfd.kind+'/samevidfr_apt36k.pkl', 'rb') as pickle_file:
                samevidframes=pickle.load(pickle_file)
        
        df = pd.DataFrame(imgs).T
        for image_id, image in imgs.items():
            if computesvf:
                print('Mapping Frame to videos...{}'.format(image_id), end= '\r')
            if aif and clean_data_path:
                txt = os.sep.join(image['file_name'].split(os.path.dirname(aif))[-1].split('\\')[2:])
                txtclean = "".join([x for x in txt if x.isascii()])
                image['file_name'] = txtclean
            file_name = image['file_name']
            if computesvf:
                imid = image['id']
                assert imid==image_id
                vid_id = image['video_id']
                samevidfr = df.index[df['video_id'] == vid_id].tolist()
                if imid in samevidfr:
                    samevidfr.remove(imid)
                samevidframes[image_id] = samevidfr
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        os.makedirs('datacache/'+cfd.kind+'',exist_ok=True)
        if not os.path.exists('datacache/'+cfd.kind+'/samevidfr_apt36k.pkl'):
            with open('datacache/'+cfd.kind+'/samevidfr_apt36k.pkl', 'wb') as pickle_file:
                pickle.dump(samevidframes, pickle_file)

        return id2name, name2id, samevidframes
    

    def __getitem__(self, index):
        results = copy.deepcopy(self.db[index])
        assert len(results)==self.num_all_frames
        # denorm = mut.Denormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # for xx in results: print(xx['image_file'],xx['bbox_id'])
        data_dict= {}
        idx_ctr = 0
        for result in results:
            result['ann_info'] = self.ann_info
            result['bbox_tfm'] = mut.xywh_to_xyxy(result['bbox']).reshape(-1,2)
            data = self.pipeline(result)
            result['bbox_tfm'] = torchvision.ops.box_convert(torch.Tensor(result['bbox_tfm']) , in_fmt='xyxy', out_fmt='xywh')
            if idx_ctr< self.num_train_frames:
                key = 'train_'
            else:
                key = 'test_'
            data_dict.setdefault(key+'images',[])
            data_dict.setdefault(key+'bbox',[])
            data_dict.setdefault(key+'skeleton',[])
            data_dict.setdefault(key+'kpts',[])
            data_dict.setdefault(key+'label',[])
            data_dict.setdefault(key+'kpts_label',[])
            data_dict.setdefault(key+'ltrb_target',[])
            data_dict.setdefault(key+'kpts_target',[])
            data_dict.setdefault(key+'target',[])
            data_dict.setdefault(key+'target_weight',[])
            # data_dict.setdefault(key+'img_metas',[])
            data_dict[key+'images'].append(data['img'])
            data_dict[key+'target'].append(torch.Tensor(data['target']))
            data_dict[key+'target_weight'].append(torch.Tensor(data['target_weight']))
            # data_dict[key+'img_metas'].append(data['img_metas'].data)
            data_dict[key+'bbox'].append(torch.Tensor(result['bbox_tfm']))
            data_dict[key+'skeleton'].append(torch.Tensor(result['ann_info']['skeleton']))
            data_dict[key+'kpts'].append(torch.Tensor(result['joints_3d'][:,:-1]))
            idx_ctr+=1
        
        for key in ['train_','test_']:
            data_dict[key+'images'] = torch.stack(data_dict[key+'images'],axis=0)
            data_dict[key+'skeleton'] = torch.stack(data_dict[key+'skeleton'],axis=0)
            data_dict[key+'target'] = torch.stack(data_dict[key+'target'],axis=0)
            data_dict[key+'target_weight'] = torch.stack(data_dict[key+'target_weight'],axis=0)
            data_dict[key+'bbox'] = torch.stack(data_dict[key+'bbox'],axis=0)
            data_dict[key+'kpts'] = torch.stack(data_dict[key+'kpts'],axis=0)
            data_dict[key+'label'] = self._generate_label_function(data_dict[key+'bbox'])
            data_dict[key+'kpts_label'] = self._generate_kptlabel_function(data_dict[key+'kpts'])
            data_dict[key+'ltrb_target'], data_dict[key+'sample_region'] = self._generate_ltbr_regression_targets(data_dict[key+'bbox'])
            data_dict[key+'kpts_target'] = self._generate_kpts_regression_targets(data_dict[key+'kpts'])
            data_dict[key+'kpts_weight'] = self._generate_kpts_vis_weight(data_dict[key+'kpts'])
            
        data_dict = TensorDict(data_dict)
        return data_dict
    
    def _generate_ltbr_regression_targets(self, target_bb):
        shifts_x = torch.arange(0, self.output_sz, step=self.stride,dtype=torch.float32, device=target_bb.device)
        shifts_y = torch.arange(
            0, self.output_sz, step=self.stride,
            dtype=torch.float32, device=target_bb.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + self.stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([target_bb[:, 0], target_bb[:, 1], target_bb[:, 0] + target_bb[:, 2],
                            target_bb[:, 1] + target_bb[:, 3]], dim=1)
        
        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        if self.use_normalized_coords:
            reg_targets_per_im = reg_targets_per_im / self.output_sz

        if self.center_sampling_radius > 0:
            is_in_box = self._compute_sampling_region(xs, xyxy, ys)
        else:
            is_in_box = (reg_targets_per_im.min(dim=1)[0] > 0)

        sz = self.output_sz//self.stride
        nb = target_bb.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)
        is_in_box = is_in_box.reshape(sz, sz, nb, 1).permute(2, 3, 0, 1)

        return reg_targets_per_im, is_in_box
    
    def _compute_sampling_region(self, xs, xyxy, ys):
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
        xmin = cx - self.center_sampling_radius * self.stride
        ymin = cy - self.center_sampling_radius * self.stride
        xmax = cx + self.center_sampling_radius * self.stride
        ymax = cy + self.center_sampling_radius * self.stride
        center_gt = xyxy.new_zeros(xyxy.shape)
        center_gt[:, 0] = torch.where(xmin > xyxy[:, 0], xmin, xyxy[:, 0])
        center_gt[:, 1] = torch.where(ymin > xyxy[:, 1], ymin, xyxy[:, 1])
        center_gt[:, 2] = torch.where(xmax > xyxy[:, 2], xyxy[:, 2], xmax)
        center_gt[:, 3] = torch.where(ymax > xyxy[:, 3], xyxy[:, 3], ymax)
        left = xs[:, None] - center_gt[:, 0]
        right = center_gt[:, 2] - xs[:, None]
        top = ys[:, None] - center_gt[:, 1]
        bottom = center_gt[:, 3] - ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_box = center_bbox.min(-1)[0] > 0
        return is_in_box
    
    def _generate_kpts_vis_weight(self,target_kp):
        weights = torch.ones(*target_kp.shape[:-1],2).to(target_kp.device)
        weights[target_kp<=0] = 0
        weights[target_kp>=self.output_sz] = 0
        weights[(weights==0).any(dim=-1)] = 0
        weights = weights.reshape(target_kp.shape[0],-1)
        return weights

    def _generate_kpts_regression_targets(self, target_kpts):
        shifts_x = torch.arange(0, self.output_sz, step=self.stride,dtype=torch.float32, device=target_kpts.device)
        shifts_y = torch.arange(
            0, self.output_sz, step=self.stride,
            dtype=torch.float32, device=target_kpts.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + self.stride // 2
        xs, ys = locations[:, 0], locations[:, 1]
        ns = target_kpts.shape[0]
        l = [xs[:, None] - x[None] for x in target_kpts[:,:,0]]
        t = [ys[:, None] - y[None] for y in target_kpts[:,:,1]]
        lt=[]
        for lx,ty in zip(l,t):
            lt.append(lx)
            lt.append(ty)
        reg_targets_per_im = torch.stack(lt, dim=2).reshape(-1, 2)
        if self.use_normalized_coords:
            reg_targets_per_im = reg_targets_per_im / self.output_sz

        sz = self.output_sz//self.stride
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, ns, -1).permute(2, 3, 0, 1)
        return reg_targets_per_im
    
    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4),
                                                      self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))

        return gauss_label
    
    def _generate_kptlabel_function(self, target_kp, bb=None):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        glbkp = []
        for kpid in range(target_kp.shape[1]):
            cxy = target_kp[:,kpid]
            gauss_label = prutils.gaussian_label_function_given_cxy(cxy,
                                                        self.label_function_params['sigma_factor'],
                                                        self.label_function_params['kernel_sz'],
                                                        self.label_function_params['feature_sz'], self.output_sz,
                                                        end_pad_if_even=self.label_function_params.get(
                                                            'end_pad_if_even', True))
            glbkp.append(gauss_label)
        gauss_label_kp =  torch.stack(glbkp,axis=1)
        return gauss_label_kp

    @staticmethod
    def center_scale_to_xyxy(center_x, center_y, width, height):
        min_x = center_x - width / 2
        min_y = center_y - height / 2
        max_x = center_x + width / 2
        max_y = center_y + height / 2
        return np.array((min_x, min_y, max_x, max_y))
    
    @staticmethod
    def xywh_to_xyxy(bbox):
        x, y, width, height = bbox
        min_x = x
        min_y = y
        max_x = x + width
        max_y = y + height
        return np.array((min_x, min_y, max_x, max_y))
    
