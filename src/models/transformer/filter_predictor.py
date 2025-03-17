import torch
import torch.nn as nn
from src.models.transformer.position_encoding import PositionEmbeddingSine
import pdb

def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FilterPredictor(nn.Module):
    def __init__(self, transformer, feature_sz, use_test_frame_encoding=True,use_kpts_enc=True,cfg=None):
        super().__init__()
        self.cfg=cfg
        self.transformer = transformer
        self.feature_sz = feature_sz
        self.use_test_frame_encoding = use_test_frame_encoding
        self.box_encoding = MLP([4, self.transformer.d_model//4, self.transformer.d_model, self.transformer.d_model])
        if use_kpts_enc == 'kom':
            print('Adding Kpts Offset Map')
            self.kpts_encoding = MLP([cfg.num_kpts*2, self.transformer.d_model//4, self.transformer.d_model, self.transformer.d_model])  # APT : 17 kpts xy-->34
        self.use_kpts_enc=use_kpts_enc
        nkpts = self.cfg.num_kpts
        if self.use_kpts_enc:
            # Reshape kpts_label from 72-->18
            self.lrsz = nn.Sequential(
                nn.Conv2d(nkpts, nkpts, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.GroupNorm(1, nkpts),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(nkpts, nkpts, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.GroupNorm(1, nkpts),
                nn.ReLU(inplace=True),
            )

        self.query_embed_fg = nn.Embedding(1, self.transformer.d_model)
        if cfg.get('use_kpts_embedding',False):
            self.query_embed_kpfg = nn.Embedding(cfg.num_kpts, self.transformer.d_model)
        if self.use_test_frame_encoding:
            self.query_embed_test = nn.Embedding(1, self.transformer.d_model)
        self.query_embed_fg_decoder = self.query_embed_fg

        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=self.transformer.d_model//2, sine_type='lin_sine',
                                                  avoid_aliazing=True, max_spatial_resolution=feature_sz)

    def forward(self, train_feat, test_feat, train_label, train_kpts_label, train_ltrb_target, train_kpts_target, *args, **kwargs):
        return self.predict_filter(train_feat, test_feat, train_label, train_kpts_label, train_ltrb_target, train_kpts_target, *args, **kwargs)

    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)
        return pos.reshape(nframes, nseq, -1, h, w)

    def predict_filter(self, train_feat, test_feat, train_label, train_kpts_label, train_ltrb_target, train_kpts_target, *args, **kwargs):
        #train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        test_pos = self.get_positional_encoding(test_feat) # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat) # Nf_tr, Ns, C, H, W

        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        # if self.use_kpts_enc and len(train_kpts_label)>0:
        if self.use_kpts_enc:
            if train_kpts_label.dim() == 5:
                _,ns,_,_,_ = train_kpts_label.shape
                train_kpts_label = train_kpts_label.reshape(-1, *train_kpts_label.shape[-3:])
                train_kpts_label = self.lrsz(train_kpts_label)
                train_kpts_label = train_kpts_label.reshape(-1, ns, *train_kpts_label.shape[-3:])
            train_kpts_label_seq_lst = [tklb.permute(1,0,2,3).flatten(1).permute(1,0).unsqueeze(2) for tklb in train_kpts_label.permute(2,0,1,3,4)]
            if self.use_kpts_enc=='kom':
                train_kpts_target_seq_T = train_kpts_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,2,Nf_tr*H*W
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C

        if self.use_kpts_enc and len(train_kpts_label)>0:
            if self.cfg.get('use_kpts_embedding',False):
                kpfg_token = self.query_embed_kpfg.weight.reshape(self.cfg.num_kpts, 1, -1)
                assert len(train_kpts_label_seq_lst)== kpfg_token.shape[0]
                train_kpts_label_enc= [x[None]* tkplbseq for x,tkplbseq in zip(kpfg_token,train_kpts_label_seq_lst)]
            else:
                train_kpts_label_enc= [fg_token* tkplbseq for tkplbseq in train_kpts_label_seq_lst]
            if self.use_kpts_enc=='kom':
                train_kpts_target_enc = self.kpts_encoding(train_kpts_target_seq_T).permute(2,0,1)

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            if self.use_kpts_enc=='kom':
                feat = torch.cat([train_feat_seq + train_label_enc + sum(train_kpts_label_enc) + train_ltrb_target_enc + train_kpts_target_enc, test_feat_seq + test_label_enc], dim=0)
            elif self.use_kpts_enc=='gmsp':
                if len(train_kpts_label)>0:
                    feat = torch.cat([train_feat_seq + train_label_enc + sum(train_kpts_label_enc) + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
                else:
                    feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)

            else:
                feat = torch.cat([train_feat_seq + train_label_enc +  train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        
        pos = torch.cat([train_pos, test_pos], dim=0)
        output_embed, enc_mem = self.transformer(feat, mask=None, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)

        enc_opt = enc_mem[-h*w:].transpose(0, 1)
        # enc_opt = enc_mem[h*w:].transpose(0, 1)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)
        return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape)
        # return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape[1:])

    def predict_cls_bbreg_filters_parallel(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        # train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        H, W = train_feat.shape[-2:]

        train_feat_stack = torch.cat([train_feat, train_feat], dim=1)
        test_feat_stack = torch.cat([test_feat, test_feat], dim=1)
        train_label_stack = torch.cat([train_label, train_label], dim=1)
        train_ltrb_target_stack = torch.cat([train_ltrb_target, train_ltrb_target], dim=1)

        test_pos = self.get_positional_encoding(test_feat)  # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat)  # Nf_tr, Ns, C, H, W

        test_feat_seq = test_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_tr*H*W, Ns, C
        train_label_seq = train_label_stack.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)  # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target_stack.permute(1, 2, 0, 3, 4).flatten(2)  # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)  # Nf_tr*H*H,Ns,C

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)

        pos = torch.cat([train_pos, test_pos], dim=0)

        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames*H*W:-h*w] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask,
                                                 query_embed=self.query_embed_fg_decoder.weight,
                                                 pos_embed=pos)

        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape(test_feat_stack.shape)
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(test_feat_stack.shape[1], -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)

        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt
