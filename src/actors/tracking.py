import os
from . import BaseActor
import torch
import pdb
# from src.utils import dcf
import torchvision
from torchvision.utils import save_image, make_grid, draw_segmentation_masks, draw_bounding_boxes, draw_keypoints
import src.utils.utilfunc as mut
import torch.nn.functional as F


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, cfg,loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.settings = cfg.data.settings
        self.cfg = cfg

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]

        # g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        # p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        g = ltrb_gth.reshape(1,n,4,-1)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.reshape(1,n,4,-1)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center
    
    def compute_iou_at_max_score_poskpts_mod(self, scores, xy_gth, xy_pred,wrs):
        xygt = xy_gth.permute(2,0,1,3,4).reshape(-1,2,*xy_gth.permute(2,0,1,3,4).shape[1:])
        xypred = xy_pred.permute(2,0,1,3,4).reshape(-1,2,*xy_pred.permute(2,0,1,3,4).shape[1:])
        tloss=0
        for tskp,prkp,gtkp,weight in zip(scores,xypred,xygt,wrs):
            ns = tskp.shape[1]
            nf = tskp.shape[0]
            ids = tskp.reshape(nf, ns, -1).max(dim=2)[1]
            fg = gtkp.reshape(2,nf,ns,-1).permute(1,2,0,3)[0, torch.arange(0, ns), :, ids]
            fp = prkp.reshape(2,nf,ns,-1).permute(1,2,0,3)[0, torch.arange(0, ns), :, ids]
            mse_loss = F.mse_loss(fp,fg,reduction='none')
            weighted_mse =  mse_loss * weight[:,:,0,None]
            tloss += torch.mean(weighted_mse)
        return tloss
    
    def max_score_no_overlap(self, scores, xy_gth, xy_pred,wrs,thresh=1e-3):
        xygt = xy_gth.permute(2,0,1,3,4).reshape(-1,2,*xy_gth.permute(2,0,1,3,4).shape[1:])
        xypred = xy_pred.permute(2,0,1,3,4).reshape(-1,2,*xy_pred.permute(2,0,1,3,4).shape[1:])
        tloss=0
        allfp= []
        for tskp,prkp,gtkp,weight in zip(scores,xypred,xygt,wrs):
            ns = tskp.shape[1]
            nf = tskp.shape[0]
            ids = tskp.reshape(nf, ns, -1).max(dim=2)[1]
            fg = gtkp.reshape(2,nf,ns,-1).permute(1,2,0,3)[0, torch.arange(0, ns), :, ids]
            fp = prkp.reshape(2,nf,ns,-1).permute(1,2,0,3)[0, torch.arange(0, ns), :, ids]
            allfp.append(fp)
        allfp = torch.stack(allfp)
        allfp= allfp.permute(2,0,1,3).squeeze(2)
        pairwise_distances = torch.cdist(allfp, allfp, p=2)
        numkps= allfp.shape[1]
        mask = 1 - torch.eye(numkps)
        mask = mask.view(1, numkps, numkps).to(allfp.device)
        pairwise_distances_no_diag = pairwise_distances * mask
        loss = torch.where(pairwise_distances_no_diag < thresh, (thresh - pairwise_distances_no_diag) ** 2, torch.tensor(0.0).to(allfp.device)).mean()
        return loss


    def compute_iou_at_max_score_pos_mod(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        ns = scores.shape[1]
        nf = scores.shape[0]
        ids = scores.reshape(nf, ns, -1).max(dim=2)[1]

        # g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        # p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        g = ltrb_gth.reshape(nf,ns,4,-1)[0, torch.arange(0, ns), :, ids].view(nf, ns, 4, 1, 1)
        p = ltrb_pred.reshape(nf,ns,4,-1)[0, torch.arange(0, ns), :, ids].view(nf, ns, 4, 1, 1)
        _, ious_pred_center = self.objective['giou'](p, g) # nf x ns x x 4 x h x w
        ious_pred_center[g.view(-1, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center
    
    

    def __call__(self, data,index):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        if self.cfg.data.settings.use_kpts_enc == 'kom':
            target_scores, bbox_preds, kp_preds, target_kpts_scores, rdict = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_bb=data['train_bbox'],
                                                train_label=data['train_label'],
                                                # train_kpts_label=data['train_kpts_label'],
                                                train_kpts_label=None,
                                                train_ltrb_target=data['train_ltrb_target'],
                                                train_kpts_target= data['train_kpts_target'],
                                                train_target= data['train_target'],
                                                train_target_weight = data['train_target_weight'],
                                                )
        else:
            target_scores, bbox_preds, kp_preds, target_kpts_scores, rdict = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_bb=data['train_bbox'],
                                                train_label=data['train_label'],
                                                train_kpts_label=None,
                                                train_ltrb_target=data['train_ltrb_target'],
                                                train_kpts_target= None,
                                                train_target= data['train_target'],
                                                train_target_weight = data['train_target_weight'],
                                                )
        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        losskpmse = self.objective['kpmse'](kp_preds, data['test_kpts_target'],data['test_kpts_weight'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_bbox'])
        
        clf_loss_test_kpts = 0
        wrs = data['test_kpts_weight'].reshape(*data['test_kpts_weight'].shape[:2],-1,2).permute(2,0,1,3)
        for tskp,dkpl,weight in zip(target_kpts_scores,data['test_kpts_label'].permute(2,0,1,3,4),wrs):
            clf_loss_test_kpts += self.objective['test_clf'](tskp, dkpl, weight=weight)

        olap_loss = self.max_score_no_overlap(target_kpts_scores, data['test_kpts_target'], kp_preds,wrs,thresh=1e-3)


        loss = self.loss_weight['giou'] * loss_giou \
                + self.loss_weight['test_clf'] * clf_loss_test \
                + self.loss_weight['kpmse']* losskpmse \
                + self.loss_weight['test_kpt_clf'] * clf_loss_test_kpts \
                + self.loss_weight['olap_loss'] * olap_loss \
                + self.loss_weight['oshot_loss'] * rdict['vith_loss']
        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')
        
        ious_pred_center = self.compute_iou_at_max_score_pos_mod(target_scores, data['test_ltrb_target'], bbox_preds)
        # kpts_msc_loss = self.compute_iou_at_max_score_poskpts_mod(target_kpts_scores, data['test_kpts_target'], kp_preds,wrs)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/KPMSE': losskpmse.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou']*loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/clf_loss_test_kpts': clf_loss_test_kpts.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf']*clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item(),
                 'olap_loss': olap_loss.item()*100,
                 'oshot_loss': rdict['vith_loss'].item() }

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious>0].std().item()

        if (index-1)% self.cfg.sample_interval==0:
            self.visualize_target(target_scores,bbox_preds, kp_preds, target_kpts_scores, data)
        return loss, stats
    
    def visualize_target(self,scores,bbop,kpop, target_kpts_scores, data):
        ns = scores.shape[1]
        nf = scores.shape[0]
        tlb = data['test_label']
        idsp = scores.reshape(nf, ns, -1).max(dim=2)[1]
        idsgt = tlb.reshape(nf,ns,-1).max(dim=2)[1]
        # max_score, max_disp = dcf.max2d(scores)
        tfs = self.settings.feature_sz
        stride = self.settings.stride
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
        name = 'image.png'

        # bbgt --> recon from ltrb
        bbgt = gtbb_xywh.view(-1,gtbb_xywh.shape[-1])
        test_images = data['test_images']
        test_bbox = data['test_bbox']
        unnorm = mut.Denormalize(mean=self.settings.normalize_mean, std=self.settings.normalize_std)
        xx= unnorm(test_images.view(-1,*test_images.shape[2:])).to(torch.uint8)
        # xx= (unnorm(test_images.view(-1,*test_images.shape[2:]))*255).to(torch.uint8)

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

        imkpp_g = [draw_keypoints(im, kpim[None], colors="green", connectivity=sk,radius=8, width=8) for im,kpim,sk in zip(xx,kpv,connvis)]


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
            # xx= (unnorm(tim.view(-1,*tim.shape[2:]))*255).to(torch.uint8)
            
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
        # kpb = torch.stack(imkpp_b,axis=0)/255 *2 -1
        # save_image(kpb, os.path.join(output_dir,name), nrow=1, normalize=True, range=(-1, 1))
        # pdb.set_trace()
        
        xy_rg = torch.stack(xy_rg,axis=0)/255 *2 -1
        xy_b = torch.stack(xy_b,axis=0)/255 *2 -1
        kpg = torch.stack(imkpp_g,axis=0)/255 *2 -1
        kpr = torch.stack(imkpp_r,axis=0)/255 *2 -1
        kpb = torch.stack(imkpp_b,axis=0)/255 *2 -1

        imtrainvis = [torch.stack(x,axis=0)/255 *2 -1 for x in ims_train]

        # catlist = [xy_rg,xy_b,kpg,kpr,kpb]
        plot_batches = self.cfg.data.settings.plot_batches
        catlist = [imtrainvis[0][:plot_batches],imtrainvis[1][:plot_batches],kpr[:plot_batches],kpb[:plot_batches]]
        kpim = torch.cat(catlist,axis=0)
        
        
        save_image(kpim, os.path.join(output_dir,name), nrow=plot_batches, normalize=True, range=(-1, 1))
        print('\n............Save a Sample.........\n')
        pass

        


    
