# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download

# 交叉卷积==1218
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 加权求和类
class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        # 是否加权标志位
        self.weight = weight  # apply weights boolean
        # 生成一个迭代器
        self.iter = range(n - 1)  # iter object
        if weight:
            # 将这个参数缓存起来，提高运行效率 第二个形参代表参数是自适应变化的，动态的
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights
    # 前向传播
    def forward(self, x):
        # 代表第一张图片张量矩阵
        y = x[0]  # no weight
        if self.weight:
            # 将w映射到0-1之间
            w = torch.sigmoid(self.w) * 2
            # 迭代n次
            for i in self.iter:
                # 加权求和
                y = y + x[i + 1] * w[i]
        else:
            # 如果加权标志为为false，就不进行加权，直接图片张量相加
            for i in self.iter:
                y = y + x[i + 1]
        return y


# 混合卷积类
class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        # 获取卷积核数量，这里属于尺寸为（1,3）卷积核
        n = len(k)  # number of convolutions
        # 通道数均衡化标志位
        if equal_ch:  # equal c_ per group
            # 以c2为步长进行划分，生成tensor列表
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            # 统计交叉通道数
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            # 生成1✖️n的元素均为c2的一维矩阵
            b = [c2] + [0] * n
            # 生成对角阵，形状为（n+1,n）,上对角线为1的矩阵
            a = np.eye(n + 1, n, k=-1)
            # 将纵轴方向的数据最后一个数据移动最前面
            a -= np.roll(a, 1, axis=1)
            # 将a与卷积核矩阵的平方进行相乘
            a *= np.array(k) ** 2
            # 将矩阵a第一行全部置1
            a[0] = 1
            # 求解最小的一元回归参数
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b
        # 模块列表
        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        # 将输出通道进行批量归一化
        self.bn = nn.BatchNorm2d(c2)
        # 激活函数定义
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

# 整体类
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            # 将提取到指定属性的对象放到指定容器y中
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        # 进行连接操作
        y = torch.cat(y, 1)  # nms ensemble
        # 返回输出
        return y, None  # inference, train output

#下载类===在进行检测或者训练之前进行一些权重文件的下载
def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    # 加载定制化的模型
    model = Ensemble()
    # 判断权重是否为列表类型，否则转为列表类型
    for w in weights if isinstance(weights, list) else [weights]:
        # 加载权重参数并下载
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        # 获取ckpt对象中的相应参数
        ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
        # 将未融合模型进行融合
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    # 迭代模型类型的各个模块，进行识别并且进行更新
    for m in model.modules():
        # 判断m的类型
        t = type(m)
        # 判断属于哪一种类型函数，有的是损失函数，有的是检测类，有的是模型类
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            # 是否替换标志位
            m.inplace = inplace  # torch 1.7.0 compatibility
            # 如果是检测类对象
            if t is Detect:
                # 判断其中anchor_grid是否为列表类型
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    # 删除对象中该属性
                    delattr(m, 'anchor_grid')
                    # 重新设置该属性
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        # 如果是Conv类型则修改相应属性
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        #     如果是上采样类，并且没有相应属性，则将对应影响因子置空
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    # 如果只有一个模型，就返回该模型
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        # 遍历该属性
        for k in ['names']:
            # 设置属性
            setattr(model, k, getattr(model[-1], k))
        #  获取模型最大的stride参数
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        # 返回模型
        return model  # return ensemble
