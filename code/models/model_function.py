
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('tools')
import parse, py_op
args = parse.args


def time_encoding_data(d = 512, time = args.time_range):
    time = int(time)
    d = int(d)
    vec = np.array([np.arange(time + 100) * int(i) for i in range(int(d/2))], dtype=np.float32).transpose()
    vec = vec / vec.max() / 2
    encoding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    encoding = torch.from_numpy(encoding)
    return encoding

