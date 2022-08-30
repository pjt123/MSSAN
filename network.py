import paddle.nn as nn
import paddle
from paddle.utils.download import get_weights_path_from_url
import math
import paddle.vision.models.resnet
# import torch.utils.model_zoo as model_zoo

import paddle.nn.functional as F

# from torch.autograd import Variable
# import torch
import lmmd
import numpy as np

__all__ = ['ResNet', 'resnet50', 'resnet18']


model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
}

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int, optional): layers of resnet, Default: 50.
        width (int, optional): base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool, optional): use pool before the last fc layer or not. Default: True.
        groups (int, optional): number of groups for each convolution block, Default: 1.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock

            # build ResNet with 18 layers
            resnet18 = ResNet(BasicBlock, 18)

            # build ResNet with 50 layers
            resnet50 = ResNet(BottleneckBlock, 50)

            # build Wide ResNet model
            wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)

            # build ResNeXt model
            resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)

            x = paddle.rand([1, 3, 224, 224])
            out = resnet18(x)

            print(out.shape)
            # [1, 1000]

    """

    def __init__(self,
                 block,
                 depth=50,
                 width=64,
                 num_classes=1000,
                 with_pool=True,
                 groups=1):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer))

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

        return x

def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def resnet18(pretrained=False, **kwargs):
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)

class ADDneck(nn.Layer):

    def __init__(self, inplanes, planes, stride=1, downsample=None,):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out





# class LambdaSheduler(nn.layer):
#     def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
#         super(LambdaSheduler, self).__init__()
#         self.gamma = gamma
#         self.max_iter = max_iter
#         self.curr_iter = 0
#
#     def lamb(self):
#         p = self.curr_iter / self.max_iter
#         lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
#         return lamb
#
#     def step(self):
#         self.curr_iter = min(self.curr_iter + 1, self.max_iter)




def focal_loss(preds, labels):
    """
          preds:softmax输出结果
          labels:真实值
          """

    weight = paddle.to_tensor([1,1,1,1,1,1,1])
    gamma = 2
    eps = 1e-7
    y_pred = preds.reshape((preds.shape[0], preds.shape[1]))  # B*C*H*W->B*C*(H*W)
    target = F.one_hot(labels, num_classes=7)

    # target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

    ce = -1 * paddle.log(y_pred + eps) * target
    floss = paddle.pow((1 - y_pred), gamma) * ce
    floss = paddle.multiply(paddle.to_tensor(floss, dtype='int64'), weight)
    floss = paddle.sum(floss, axis=1)
    return paddle.mean(floss)


class MFSAN(nn.Layer):

    def __init__(self, num_classes=7):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet18(True)
        self.sonnet1 = ADDneck(512, 256)
        self.sonnet2 = ADDneck(512, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2D(7, stride=1)
        # self.focalloos = Focal_Loss()

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        lmmd_loss = 0

        if self.training == True:
            if mark == 1:
                data_src = self.sharedNet(data_src)   # [32, 2048,7,7]
                data_tgt = self.sharedNet(data_tgt)   # [32, 2048,7,7]

                data_tgt_son1 = self.sonnet1(data_tgt)   # [32, 256,7,7]
                data_tgt_son1 = self.avgpool(data_tgt_son1)  # [32, 256,1,1]
                # print(data_tgt_son1.shape[0])
                data_tgt_son1 = paddle.reshape(data_tgt_son1, [data_tgt_son1.shape[0], -1])     # [32, 256]

                data_tgt_clf1 = self.cls_fc_son1(data_tgt_son1)    # [32,7]
                data_tgt_logits1 = paddle.nn.functional.softmax(data_tgt_clf1, axis=1)    # [32,7]

                data_src = self.sonnet1(data_src)    # [32, 256,7,7]
                data_src = self.avgpool(data_src)    # [32, 256,1,1]
                data_src = paddle.reshape(data_src, [data_src.shape[0], -1])     # [32, 256]

# =============================================lmmd与adv损失===============================

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = paddle.reshape(data_tgt_son2, [data_tgt_son2.shape[0], -1])
                # adv_loss += self.domain_classifier_1(data_tgt_son1, data_tgt_son2)

                data_tgt_clf2 = self.cls_fc_son2(data_tgt_son2)
                data_tgt_logits2 = paddle.nn.functional.softmax(data_tgt_clf2, axis=1)

                data_tgt_logits = (data_tgt_logits1 + data_tgt_logits2) / 2
                lmmd_loss += lmmd.lmmd(data_src, data_tgt_son1, label_src, data_tgt_logits)
#  ====================================================l1损失====================================================
                l1_loss = paddle.abs(data_tgt_logits1 - data_tgt_logits2)
                l1_loss = paddle.mean(l1_loss)

# ===============================================L2============================
#                 l2_loss = F .mse_loss(data_tgt_logits1, data_tgt_logits2)

                pred_src = self.cls_fc_son1(data_src)
                # print(F.log_softmax(pred_src, axis=1))
                # print(paddle.to_tensor(label_src, dtype='int64'))
                # cls_loss = F.nll_loss(F.log_softmax(pred_src, axis=1), paddle.to_tensor(label_src, dtype='int64'))
                cls_loss = focal_loss(F.softmax(pred_src, axis=1), paddle.to_tensor(label_src, dtype='int64'))

                return cls_loss, lmmd_loss, l1_loss                            # l1_loss

            if mark == 2:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = paddle.reshape(data_tgt_son2, [data_tgt_son2.shape[0], -1])
                data_tgt_clf2 = self.cls_fc_son2(data_tgt_son2)
                data_tgt_logits2 = paddle.nn.functional.softmax(data_tgt_clf2, axis=1)
                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = paddle.reshape(data_src, [data_src.shape[0], -1])

# =============================================lmmd与adv损失===============================


                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = paddle.reshape(data_tgt_son1, [data_tgt_son1.shape[0], -1])
                # adv_loss += self.domain_classifier_1(data_tgt_son1, data_tgt_son2)

                data_tgt_clf1 = self.cls_fc_son1(data_tgt_son1)
                data_tgt_logits1 = paddle.nn.functional.softmax(data_tgt_clf1, axis=1)

                data_tgt_logits = (data_tgt_logits1 + data_tgt_logits2) / 2
                lmmd_loss += lmmd.lmmd(data_src, data_tgt_son2, label_src, data_tgt_logits)

#  ====================================================l1损失====================================================
                l1_loss = paddle.abs(data_tgt_logits1 - data_tgt_logits2)
                l1_loss = paddle.mean(l1_loss)

# ===============================================L2============================
#                 l2_loss = F.mse_loss(data_tgt_logits1, data_tgt_logits2)

                pred_src = self.cls_fc_son2(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, axis=1), paddle.to_tensor(label_src, dtype='int64'))
                # cls_loss = focal_loss(F.softmax(pred_src, dim=1), label_src)

                return cls_loss, lmmd_loss, l1_loss      # l1_loss

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = paddle.reshape(fea_son1, [fea_son1.shape[0], -1])
            pred1 = self.cls_fc_son1(fea_son1)
            # adv_loss_1 = self.domain_classifier_1(fea_son1, fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = paddle.reshape(fea_son2, [fea_son2.shape[0], -1])
            pred2 = self.cls_fc_son2(fea_son2)
            # adv_loss_2, _ = self.domain_classifier_2(fea_son2, fea_son2)
            return pred1, pred2

