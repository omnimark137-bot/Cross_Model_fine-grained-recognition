
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
	$ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import math

# 修正为相对import
from .models.common import *
from .models.experimental import *
from .utils.autoanchor import check_anchor_order
from .utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from .utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
							   time_sync)
from .utils.global_var import init
from .models.hrnet import *
from .models.transpose_r import *
from .utils.global_var import set_value, get_value
try:
	import thop  # for FLOPs computation
except ImportError:
	thop = None
from copy import deepcopy


# YOLO/PGD Detect 主干类（简化版，按常见实现补全）
class Detect(nn.Module):
	def __init__(self, nc=80, anchors=(), ch=()):
		super().__init__()
		self.nc = nc  # 类别数
		self.no = nc + 5  # 每个锚框输出（类别+xywh+obj）
		self.nl = len(anchors)  # 检测层数
		self.na = len(anchors[0]) // 2 if anchors else 3  # 每层锚框数
		self.grid = [torch.empty(0) for _ in range(self.nl)]
		self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
		self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
		self.m = nn.ModuleList([nn.Conv2d(x, self.no * self.na, 1) for x in ch])

	def forward(self, x):
		z = []
		for i in range(self.nl):
			x[i] = self.m[i](x[i])
			bs, _, ny, nx = x[i].shape
			x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
			z.append(x[i])
		return z
