# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on code from Jianwei Yang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.utils.config import cfg
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def init_conv(conv,glu=True):
  init.xavier_uniform(conv.weight)
  if conv.bias is not None:
    conv.bias.data.zero_()

def init_linear(linear):
  init.constant(linear.weight,0)
  init.constant(linear.bias, 1)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False,meta_train=True,meta_test=None,meta_loss=None):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.meta_train = meta_train
    self.meta_test = meta_test
    self.meta_loss = meta_loss
    self.proto_dim = cfg.PROTO.DIM
    self.ang_dim = cfg.PROTO.D_THETA
    self.lambda_dec = cfg.PROTO.LAMBDA_DEC
    self.lambda_fg = cfg.PROTO.LAMBDA_FG

    _fasterRCNN.__init__(self, classes, class_agnostic,meta_train,meta_test,meta_loss)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.meta_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.rcnn_conv1 = resnet.conv1

    self.RCNN_base = nn.Sequential(resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.sigmoid = nn.Sigmoid()
    self.max_pooled = nn.MaxPool2d(2)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)

    if self.meta_loss:
      self.Meta_cls_score = nn.Linear(2048, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4) # x,y,w,h
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # prototype learners
    self.fc_proto = nn.Linear(2048, self.proto_dim)
    self.conv_angle = nn.Conv2d(self.dout_base_model, self.ang_dim, kernel_size=3, stride=1, padding=1, bias=True)
    self.fc_ang_proj = nn.Linear(self.ang_dim, self.proto_dim)
    self.fc_att = nn.Linear(self.proto_dim, 2048)
    self.proto_cls_score = nn.Linear(2048, 1)


    # Fix blocks
    for p in self.rcnn_conv1.parameters(): p.requires_grad=False
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False


    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 5)
    if cfg.RESNET.FIXED_BLOCKS >= 4:
      for p in self.RCNN_top.parameters(): p.requires_grad = False
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[3].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[4].train()
      self.RCNN_base[5].train()

      self.RCNN_base.eval()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def apply_prototype_attention(self, pooled_feat, prototypes=None):
    base_cls_score = self.RCNN_cls_score(pooled_feat)
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    if prototypes is None or prototypes.get('P_pure') is None:
      return base_cls_score, bbox_pred

    P_pure = prototypes['P_pure']
    class_ids = prototypes.get('class_ids', list(range(P_pure.size(0))))
    alpha = torch.sigmoid(self.fc_att(P_pure))
    alpha = alpha.unsqueeze(0)  # 1 x C x D
    roi = pooled_feat.unsqueeze(1)  # N x 1 x D
    weighted = roi * alpha  # N x C x D
    logits = self.proto_cls_score(weighted).squeeze(-1)  # N x C

    proto_scores = base_cls_score.clone()
    for idx, cid in enumerate(class_ids):
      if cid < proto_scores.size(1):
        proto_scores[:, cid] += logits[:, idx]
    return proto_scores, bbox_pred

  def prn_network(self, im_data, prn_cls):
    '''
    Dual-branch prototype learner producing class and angle prototypes.
    '''
    if cfg.mask_on:
      base_feat = self.RCNN_base(self.rcnn_conv1(im_data))
    else:
      base_feat = self.RCNN_base(self.meta_conv1(im_data))

    pooled = self._head_to_tail(self.max_pooled(base_feat))
    cls_embed = self.fc_proto(pooled)
    angle_feat = self.conv_angle(base_feat)
    angle_desc = angle_feat.view(angle_feat.size(0), angle_feat.size(1), -1).max(dim=2)[0]

    cls_dict = {}
    ang_dict = {}
    class_ids = []
    for idx in range(len(prn_cls)):
      cls_id = int(prn_cls[idx].cpu().numpy()[0]) if isinstance(prn_cls[idx], torch.Tensor) else int(prn_cls[idx])
      class_ids.append(cls_id)
      cls_dict.setdefault(cls_id, []).append(cls_embed[idx])
      ang_dict.setdefault(cls_id, []).append(angle_desc[idx])

    proto_list = []
    ang_list = []
    order = []
    for cid in sorted(set(class_ids)):
      order.append(cid)
      proto_list.append(torch.stack(cls_dict[cid], dim=0).mean(dim=0))
      ang_list.append(torch.stack(ang_dict[cid], dim=0).mean(dim=0))

    P_init = torch.stack(proto_list, dim=0)
    P_ang = torch.stack(ang_list, dim=0)
    if cfg.PROTO.TOP_M and cfg.PROTO.TOP_M > 0:
      vals, idx = torch.topk(P_ang, cfg.PROTO.TOP_M, dim=1)
      P_ang = vals

    P_ang_tilde = self.fc_ang_proj(P_ang)
    dot = (P_init * P_ang_tilde).sum(dim=1, keepdim=True)
    denom = (P_ang_tilde.pow(2).sum(dim=1, keepdim=True) + 1e-6)
    P_pure = P_init - dot / denom * P_ang_tilde
    P_pure = P_pure / (P_pure.norm(dim=1, keepdim=True) + 1e-6)
    L_dec = ((F.cosine_similarity(P_init, P_ang_tilde, dim=1)) ** 2).mean()

    self.proto_outputs = {
      'P_init': P_init,
      'P_pure': P_pure,
      'P_ang': P_ang,
      'P_ang_tilde': P_ang_tilde,
      'class_ids': order,
      'L_dec': L_dec * self.lambda_dec
    }
    return self.proto_outputs