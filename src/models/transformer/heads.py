import importlib
import os
import torch
import torch.nn as nn
import src.models.layers.filter as filter_layer
import pdb
# from src.models.vitheads.configs.config_vithead_apt36k import vithead_config
from src.models.vitheads.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, classifier_kpt,
                 bb_regressor, kpt_regressor,
                 separate_filters_for_cls_and_bbreg=False,cfg=None):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classifier_kpt = classifier_kpt
        self.bb_regressor = bb_regressor
        self.kpt_regressor = kpt_regressor
        self.cfg = cfg
        vit_config_path = cfg.vitconfig
        dcname = vit_config_path.split(os.sep)[-1].split('.')[0]
        spec = importlib.util.spec_from_file_location(dcname, vit_config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vithead_config = getattr(module,dcname)
        self.keypoint_head = TopdownHeatmapSimpleHead(vithead_config)
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)

        ##TODO Find estimate of frame kpts here

        # Train filter
        if self.cfg.use_kpts_enc == 'gmsp':
            losses = dict()
            khv = []
            ftkp_total_loss= 0
            acc_vithead = []

            if kwargs.get('train_target',None) is not None:
                if type(kwargs.get('train_target',None)) == str:
                    print('Validation.. ')
                    for frame in train_feat:
                        ftkp = self.keypoint_head(frame)
                        khv.append(ftkp)
                    khv= torch.stack(khv)
                else:
                    for frame,tgt,tgtw in zip(train_feat,kwargs['train_target'],kwargs['train_target_weight']):
                        ftkp = self.keypoint_head(frame)
                        ftkp_total_loss += self.keypoint_head.get_loss(ftkp,tgt,tgtw)['heatmap_loss']
                        acc_vithead.append(self.keypoint_head.get_accuracy(ftkp, tgt, tgtw)['acc_pose'])
                        khv.append(ftkp)
                    khv= torch.stack(khv)
            kwargs['train_kpts_label'] = khv
            kwargs['train_kpts_target']= None

        cls_filter, cls_kpt_filter, breg_filter, kpreg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)
        target_scores = self.classifier(test_feat_enc, cls_filter)
        target_kpts_scores = self.classifier_kpt(test_feat_enc, cls_kpt_filter)


        # compute the final prediction using the output module
        test_hm = self.keypoint_head(test_feat[0])[None]
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)  # (nf*ns, 4, h, w)
        kp_preds = self.kpt_regressor(test_feat_enc, kpreg_filter,test_hm)

        # Added Later --> Verify
        bbox_preds = bbox_preds.view(target_scores.shape[0],target_scores.shape[1],*bbox_preds.shape[-3:])
        kp_preds = kp_preds.view(target_scores.shape[0],target_scores.shape[1],*kp_preds.shape[-3:])

        rdict = {
            'vith_loss' : ftkp_total_loss,  
        }
        return target_scores, bbox_preds, kp_preds, target_kpts_scores, rdict

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, train_kpts_label ,*args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, train_kpts_label, *args, **kwargs)
            cls_weights = cls_kpt_weights = bbreg_weights = kptreg_weights =  weights

        return cls_weights, cls_kpt_weights, bbreg_weights, kptreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)

class LinearFilterClassifier_KPT(nn.Module):
    def __init__(self, num_channels, num_kpts=17, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter
        self.numkpt = num_kpts

        if project_filter:
            self.linear = nn.ModuleList()
            for _ in range(self.numkpt):
                self.linear.append(nn.Linear(self.num_channels, self.num_channels))

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj= []
            for idx in range(self.numkpt):
                filter_proj.append(self.linear[idx](filter.reshape(-1, self.num_channels)).reshape(filter.shape))
        else:
            raise NotImplementedError('Modify')
            filter_proj = filter
        out= []
        for fp in filter_proj:
            out.append(filter_layer.apply_filter(feat, fp))
        return out


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb



class KPTRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True, feature_sz=18,num_kpts=17):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter
        self.feature_sz=feature_sz

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        layers = []
        layers.extend(conv_layer(num_kpts, num_channels))  # Hardcode to 17
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend([nn.Upsample(size=(self.feature_sz, self.feature_sz), mode='bilinear', align_corners=False)])
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower_heatmap = nn.Sequential(*layers)
        
        self.layer_intermediate = nn.Sequential(*conv_layer(num_channels*2, num_channels))
        self.kpreg_layer = nn.Conv2d(num_channels, num_kpts*2, kernel_size=3, dilation=1, padding=1)  # 17 kpts

    def forward(self, feat, filter, heatmap):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        hmrs = heatmap.reshape(-1, heatmap.shape[-3], heatmap.shape[-2], heatmap.shape[-1])
        hmrs=self.tower_heatmap(hmrs)
        att_feats = feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])
        feats_f=torch.cat((att_feats,hmrs),axis=1)
        feats_f= self.layer_intermediate(feats_f)
        feats_tower = self.tower(feats_f) # (nf*ns, c, h, w)
        ltrb = torch.exp(self.kpreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb

