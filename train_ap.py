import numpy as np
from omegaconf import open_dict
import pdb
import data.dataloader as dls
from src.models.tracking import tompnet
from src.models.loss.bbr_loss import GIoULoss, KpMSELoss
import src.models.loss as ltr_losses
import src.actors.tracking as actors
import torch.optim as optim
from src.trainers.ltr_trainer import LTRTrainer
from data import loader



def training_loop(cfg):
    cfd = cfg.data
    settings = cfg.data.settings
    device= 'cuda'

    with open_dict(settings):
        settings.output_sz = settings.feature_sz * settings.stride     # Size of input image crop

    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    label_params = {'feature_sz': settings.feature_sz, 
                    'sigma_factor': output_sigma, 
                    'kernel_sz': settings.target_filter_sz}
    
    dataset_train = dls.APT36Kmm(cfd,
                                samples_per_videos= settings.sample_per_videos, num_test_frames=settings.num_test_frames,
                                num_train_frames=settings.num_train_frames,
                                label_function_params=label_params,
                                stride=settings.stride)
    reload_args={}
    reload_args['reload']=True
    reload_args['dataset'] = {'dls': dls.APT36Kmm,
                                   'cfd':cfd,
                                   'samples_per_videos': settings.sample_per_videos,
                                   'num_test_frames':settings.num_test_frames,
                                    'num_train_frames':settings.num_train_frames,
                                    'label_function_params':label_params
                                   }
    loader_train = loader.LTRLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                             stack_dim=1, batch_size=settings.batch_size)
    reload_args['dataloader'] = {'loader': loader.LTRLoader,
                                   'name':'train',
                                   'dataset': None,
                                   'training':True,
                                    'num_workers':settings.num_workers,
                                    'stack_dim':1,
                                    'batch_size':settings.batch_size,
                                   }
    net = tompnet.tompnet101(filter_size=settings.target_filter_sz, backbone_pretrained=True, head_feat_blocks=0,
                             num_kpts=settings.num_kpts,
                             head_feat_norm=True, final_conv=True, out_feature_dim=256, feature_sz=settings.feature_sz,
                             frozen_backbone_layers=settings.frozen_backbone_layers,
                             num_encoder_layers=settings.num_encoder_layers,
                             num_decoder_layers=settings.num_decoder_layers,
                             use_test_frame_encoding=settings.use_test_frame_encoding,
                             separate_filters_for_cls_and_bbreg=settings.separate_filter,
                             cfg=settings)
    
    objective = {'giou': GIoULoss(),
                 'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                 'kpmse': KpMSELoss(),
                 }
    loss_weight = {'giou': settings.weight_giou, 
                   'test_clf': settings.weight_clf,
                   'kpmse': settings.weight_kpmse,
                   'test_kpt_clf': settings.weight_clf_kpt,
                   'olap_loss' : settings.weight_olap_loss,
                   'oshot_loss' : settings.weight_oshot_loss
                   }



    net = net.to(device)
    actor = actors.ToMPActor(net=net, objective=objective, cfg=cfg,loss_weight=loss_weight)
    optimizer = optim.AdamW([
        {'params': actor.net.head.parameters(), 'lr': 1e-4},
        {'params': actor.net.feature_extractor.layer3.parameters(), 'lr': 2e-5}
    ], lr=2e-4, weight_decay=0.0001)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler,
                         freeze_backbone_bn_layers=settings.freeze_backbone_bn_layers,
                         reload_data= reload_args,
                         )
    trainer.train(settings.num_epochs, load_latest=True, fail_safe=False,reload_data=False)
