from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

# 将所有类别的attentions整合成一个
# 1. 先将每一类的2048维的attention向量->2048维
# 2. 将所有降维后的向量concat(如果直接concat起来，那维度就是动态变化的了)，直接相加算了
# 3. 将concat以后的向量->1024维
class AttentionsMerge(nn.Module):
    
    def __init__(self, numOfAttentions):
        super(AttentionsMerge, self).__init__()
        self.numOfAttentions = 20
        self.d_reduction = nn.Sequential( # 第一步
            nn.Linear(2048, 1024) ,
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.merge = nn.Linear(2048, 1024) # 第三步
        self.sigmoid = nn.Sigmoid()

    def forward(self, attentions):
        if isinstance(attentions, (Variable, torch.Tensor)):
            if attentions.dim() == 1:
                attentions = [attentions]
            else:
                attentions = list(torch.unbind(attentions, dim=0))
        elif isinstance(attentions, (list, tuple)):
            attentions = list(attentions)
        else:
            raise TypeError('Unsupported attentions type: {}'.format(type(attentions)))

        if len(attentions) == 0:
            raise ValueError('attentions should not be empty when provided')

        start_idx = 1 if len(attentions) > 1 else 0
        attentionSum = attentions[start_idx].clone()
        for i in range(start_idx + 1, len(attentions)):
            attentionSum += attentions[i]

        if attentionSum.dim() == 1:
            attentionSum = attentionSum.unsqueeze(0)
        elif attentionSum.dim() > 2:
            attentionSum = attentionSum.view(-1, attentionSum.size(-1))

        mergedattention = self.merge(attentionSum)
        mergedattention = self.sigmoid(mergedattention)

        return mergedattention


# RPN 特征预测参数的部分
class Attention2Weights(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(Attention2Weights, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kerner_size = kernel_size

        self.metablock = nn.Sequential(
            nn.Linear(self.in_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_channel*512*kernel_size*kernel_size)
        )

    def forward(self, proto):
        weight = self.metablock(proto)

        return weight


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, num_classes, lambda_fg=1.0, lambda_bg=1.0, lambda_attn=1.0):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # weights for combining prototype foreground/background maps when provided
        self.lambda_fg = lambda_fg
        self.lambda_bg = lambda_bg
        self.lambda_attn = lambda_attn

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 原版metarcnn需要把下面注释掉，改进版需要解注释
        # 对attentions的预处理
        self.Merge_Attention = AttentionsMerge(num_classes)
        # 和RPN_cls_score部分构成残差
        # in_channel：attention向量的维度1024 out_channel：先试试看512
        self.Class_Attention_Conv = Attention2Weights(in_channel=1024, out_channel=self.nc_score_out, kernel_size=1)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer 
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, attentions=None, prototypes=None):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) 
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1) 

        # 原版metarcnn需要把下面注释掉，改进版需要解注释
        # 在这里加上残差模块
        if attentions is not None:
            if isinstance(attentions, (Variable, torch.Tensor)):
                attentions = list(torch.unbind(attentions, dim=0)) if attentions.dim() > 1 else [attentions]
            merged_attention = self.Merge_Attention(attentions)
            weights = self.Class_Attention_Conv(merged_attention).view(self.nc_score_out, 512, 1, 1)
            Atten_Score = nn.functional.conv2d(rpn_conv1, weights)
            rpn_cls_score = rpn_cls_score * (self.lambda_attn * Atten_Score) + rpn_cls_score

        # Optionally fuse prototype maps (foreground/background cues) if provided.
        # The prototypes are expected to be spatial maps that can be broadcast or
        # interpolated to match the RPN classification map's spatial size.
        if prototypes is not None:
            proto_fg = prototypes
            proto_bg = None

            # unpack prototype dicts that may provide explicit fg/bg maps
            if isinstance(prototypes, dict):
                proto_fg = prototypes.get('fg') or prototypes.get('proto_fg') or prototypes.get('foreground')
                proto_bg = prototypes.get('bg') or prototypes.get('proto_bg') or prototypes.get('background')
                # fall back to any single tensor value if no common key found
                if proto_fg is None and proto_bg is None:
                    for value in prototypes.values():
                        if torch.is_tensor(value):
                            proto_fg = value
                            break

            if proto_fg is None and proto_bg is None:
                # nothing usable provided
                proto_fg = None

            def _ensure_4d(tensor):
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(1)
                if tensor.dim() != 4:
                    raise ValueError('prototypes must be 3D or 4D tensor, got {} dims'.format(tensor.dim()))
                return tensor

            if proto_fg is not None:
                proto_fg = _ensure_4d(proto_fg)
            if proto_bg is not None:
                proto_bg = _ensure_4d(proto_bg)

            # map channel dimension to the expected foreground/background channel counts
            target_fg_channels = self.nc_score_out // 2
            target_bg_channels = target_fg_channels

            if proto_fg is not None:
                if proto_fg.size(1) == 1:
                    proto_fg = proto_fg.expand(proto_fg.size(0), target_fg_channels, proto_fg.size(2), proto_fg.size(3))
                elif proto_fg.size(1) != target_fg_channels:
                    proto_fg = proto_fg[:, :target_fg_channels, :, :]
                    if proto_fg.size(1) < target_fg_channels:
                        repeat_count = target_fg_channels - proto_fg.size(1)
                        pad = proto_fg[:, -1:, :, :].expand(proto_fg.size(0), repeat_count, proto_fg.size(2), proto_fg.size(3))
                        proto_fg = torch.cat([proto_fg, pad], dim=1)

            if proto_bg is not None:
                if proto_bg.size(1) == 1:
                    proto_bg = proto_bg.expand(proto_bg.size(0), target_bg_channels, proto_bg.size(2), proto_bg.size(3))
                elif proto_bg.size(1) != target_bg_channels:
                    proto_bg = proto_bg[:, :target_bg_channels, :, :]
                    if proto_bg.size(1) < target_bg_channels:
                        repeat_count = target_bg_channels - proto_bg.size(1)
                        pad = proto_bg[:, -1:, :, :].expand(proto_bg.size(0), repeat_count, proto_bg.size(2), proto_bg.size(3))
                        proto_bg = torch.cat([proto_bg, pad], dim=1)

            # resize spatially if needed
            if proto_fg is not None and proto_fg.shape[2:] != rpn_cls_score.shape[2:]:
                proto_fg = F.interpolate(proto_fg, size=rpn_cls_score.shape[2:], mode='bilinear', align_corners=False)
            if proto_bg is not None and proto_bg.shape[2:] != rpn_cls_score.shape[2:]:
                proto_bg = F.interpolate(proto_bg, size=rpn_cls_score.shape[2:], mode='bilinear', align_corners=False)

            fg_target = rpn_cls_score[:, 1::2, :, :]
            bg_target = rpn_cls_score[:, 0::2, :, :]

            if proto_fg is not None:
                proto_fg = proto_fg.expand_as(fg_target)
                rpn_cls_score[:, 1::2, :, :] = fg_target + self.lambda_fg * proto_fg
            if proto_bg is not None:
                proto_bg = proto_bg.expand_as(bg_target)
                rpn_cls_score[:, 0::2, :, :] = bg_target + self.lambda_bg * proto_bg
            elif proto_fg is not None:
                # derive background cues from foreground if explicit bg map not provided
                proto_bg = (1 - proto_fg).expand_as(bg_target)
                rpn_cls_score[:, 0::2, :, :] = bg_target + self.lambda_bg * proto_bg


        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape,dim=1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box


class _RPN_out_pred_label(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN_out_pred_label, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, rpn_cls_prob, self.rpn_loss_cls, self.rpn_loss_box