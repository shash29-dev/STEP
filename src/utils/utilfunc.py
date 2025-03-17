
import pdb
import torch.nn as nn
import cv2 as cv
from typing import Any, List, Tuple, Union
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from mmpose.datasets.pipelines import Compose as mpCompose
import torchvision
import data.processing_utils as prutils
import torch
from data.tensordict import TensorDict


def freeze_batchnorm_layers(net):
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None
    
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]



class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_tensor):
        denormalized_image = image_tensor.clone()  # Create a copy of the tensor
        for channel in range(3):  # Assuming 3-channel RGB image
            denormalized_image[:, channel, :, :] = denormalized_image[:, channel, :, :] * self.std[channel] + self.mean[channel]
        return denormalized_image
    

def xywh_to_xyxy(bbox):
    x, y, width, height = bbox
    min_x = x
    min_y = y
    max_x = x + width
    max_y = y + height
    return np.array((min_x, min_y, max_x, max_y))

def add_offset_to_bbox(bbox,offset_percentage):
    current_center_x = (bbox[0] + bbox[2]) / 2
    current_center_y = (bbox[1] + bbox[3]) / 2
    
    expand_x = bbox[2] * offset_percentage / 2
    expand_y = bbox[3] * offset_percentage / 2
    new_bbox = [
            current_center_x - bbox[2] / 2 - expand_x,  # xmin
            current_center_y - bbox[3] / 2 - expand_y,  # ymin
            bbox[2] + 2 * expand_x,  # expanded width
            bbox[3] + 2 * expand_y   # expanded height
        ]
    
    return new_bbox

def xyxy_to_xywh(xyxy_bbox):
    x1, y1, x2, y2 = xyxy_bbox
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return np.array(( x, y, width, height))

def flip_bbox_horizontal(original_bbox, image_width):
    x1, y1, x2, y2 = original_bbox

    x1_flipped = image_width - x2
    x2_flipped = image_width - x1

    flipped_bbox = np.array([x1_flipped, y1, x2_flipped, y2])
    return flipped_bbox

def center_scale_to_xyxy(bbox):
    center_x, center_y, width, height = bbox
    min_x = center_x - width / 2
    min_y = center_y - height / 2
    max_x = center_x + width / 2
    max_y = center_y + height / 2
    return np.array((min_x, min_y, max_x, max_y))

def flip_bbox_center_scale(original_bbox, image_width):
    center_x, center_y, width, height = original_bbox

    x_left = center_x - width / 2
    x_right = center_x + width / 2

    x_left_flipped = image_width - x_right
    x_right_flipped = image_width - x_left

    center_flipped = (x_left_flipped + x_right_flipped) / 2

    width_flipped = x_right_flipped - x_left_flipped

    flipped_bbox = ([center_flipped, center_y, width_flipped, height])
    return flipped_bbox


def transform_bbox_affine_matrix(original_bbox, affine_matrix):
    x1, y1, x2, y2 = original_bbox

    # Apply affine transformation to each corner
    corners = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x1, y2, 1],
        [x2, y2, 1]
    ])
    transformed_corners = np.dot(corners, affine_matrix.T)

    # Find the new minimum and maximum values
    x_min = np.min(transformed_corners[:, 0])
    x_max = np.max(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    y_max = np.max(transformed_corners[:, 1])

    # Return the transformed bounding box
    transformed_bbox = np.array([x_min, y_min, x_max, y_max])
    return transformed_bbox

def _xywh2cs(x, y, w, h, ann_info, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = ann_info['image_size'][0] / ann_info['image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)


        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        # parsc = np.random.randint(low=100,high=200)
        parsc=200
        scale = np.array([w / parsc, h / parsc], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding
        return center, scale

def get_processed(datain,pipeline,ann_info,cfd,lfp,padding =1.5):
    pipeline = mpCompose(pipeline)
    output_sz = cfd.settings.output_sz
    stride= 16
    use_normalized_coords = cfd.settings.normalized_bbreg_coords
    center_sampling_radius = cfd.settings.center_sampling_radius
    lfp = lfp
        
    vd = Valdutils(output_sz=output_sz,stride=stride,unc=use_normalized_coords,
                   csr=center_sampling_radius,
                   lfp=lfp)
    result= {}
    data_dict= {}
    for ti,tb,tkps,impath in zip(datain['train_images'],datain['train_bbox'], datain['train_kpts'],datain['train_impath']):
        key= 'train_'
        center, scale = _xywh2cs(*tb.tolist(),ann_info,padding=padding)
        result['ann_info']= ann_info
        result['img'] = ti
        result['center']=center
        result['scale']=scale
        joints_3d = np.zeros((tkps.shape[0], 3), dtype=np.float32)
        joints_3d_visible = np.zeros((tkps.shape[0], 3), dtype=np.float32)
        keypoints = tkps.numpy()
        joints_3d[:, :2] = keypoints[:, :2]
        joints_3d[:, -1] = (keypoints.sum(axis=1)!=0).astype(float)
        joints_3d_visible[:, :2] = np.minimum(1, joints_3d[:, 2:3])
        result['joints_3d']=joints_3d
        result['joints_3d_visible']=joints_3d_visible
        result['gmsp_kps_joints']= np.zeros_like(joints_3d[None])
        result['gmsp_kps_joints_visible']= np.zeros_like(joints_3d_visible[None])
        result['rotation']=0
        result['bbox_tfm'] = xywh_to_xyxy(tb.tolist()).reshape(-1,2)
        result['image_file']=impath
        result['bbox_score']=1
        data = pipeline(result)
        result['bbox_tfm'] = torchvision.ops.box_convert(torch.Tensor(result['bbox_tfm']) , in_fmt='xyxy', out_fmt='xywh')
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
        # data_dict[key+'img_metas'].append(data['img_metas'].data)
        data_dict[key+'bbox'].append(torch.Tensor(result['bbox_tfm']))
        data_dict[key+'skeleton'].append(torch.Tensor(result['ann_info']['skeleton']))
        data_dict[key+'kpts'].append(torch.Tensor(result['joints_3d'][:,:-1]))

    metas = []
    for ti,tb,tkps,impath in zip(datain['test_images'],datain['test_bbox'], datain['test_kpts'],datain['test_impath']):
        key= 'test_'
        center, scale = _xywh2cs(*tb.tolist(),ann_info,padding=padding)
        result['ann_info']= ann_info
        result['img'] = ti
        result['center']=center
        result['scale']=scale
        joints_3d = np.zeros((tkps.shape[0], 3), dtype=np.float32)
        joints_3d_visible = np.zeros((tkps.shape[0], 3), dtype=np.float32)
        keypoints = tkps.numpy()
        joints_3d[:, :2] = keypoints[:, :2]
        joints_3d[:, -1] = (keypoints.sum(axis=1)!=0).astype(float)
        joints_3d_visible[:, :2] = np.minimum(1, joints_3d[:, 2:3])
        result['joints_3d']=joints_3d
        result['joints_3d_visible']=joints_3d_visible
        result['rotation']=0
        result['bbox_tfm'] = xywh_to_xyxy(tb.tolist()).reshape(-1,2)
        result['image_file']=impath
        result['bbox_score']=1
        data = pipeline(result)
        metas.append(data['img_metas'].data)
        result['bbox_tfm'] = torchvision.ops.box_convert(torch.Tensor(result['bbox_tfm']) , in_fmt='xyxy', out_fmt='xywh')
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
        # data_dict[key+'img_metas'].append(data['img_metas'].data)
        data_dict[key+'bbox'].append(torch.Tensor(result['bbox_tfm']))
        data_dict[key+'skeleton'].append(torch.Tensor(result['ann_info']['skeleton']))
        data_dict[key+'kpts'].append(torch.Tensor(result['joints_3d'][:,:-1]))

    for key in ['train_','test_']:
        data_dict[key+'images'] = torch.stack(data_dict[key+'images'],axis=0)
        data_dict[key+'skeleton'] = torch.stack(data_dict[key+'skeleton'],axis=0)
        data_dict[key+'bbox'] = torch.stack(data_dict[key+'bbox'],axis=0)
        data_dict[key+'kpts'] = torch.stack(data_dict[key+'kpts'],axis=0)
        data_dict[key+'label'] = vd._generate_label_function(data_dict[key+'bbox'])
        data_dict[key+'kpts_label'] = vd._generate_kptlabel_function(data_dict[key+'kpts'])
        data_dict[key+'ltrb_target'], data_dict[key+'sample_region'] = vd._generate_ltbr_regression_targets(data_dict[key+'bbox'])
        data_dict[key+'kpts_target'] = vd._generate_kpts_regression_targets(data_dict[key+'kpts'])
    data_dict = TensorDict(data_dict)   
    return data_dict, metas, vd

class Valdutils(object):
    def __init__(self,output_sz,stride,unc,csr,lfp):
        self.output_sz=output_sz
        self.stride = stride
        self.use_normalized_coords = unc
        self.center_sampling_radius = csr 
        self.label_function_params = lfp

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
    
