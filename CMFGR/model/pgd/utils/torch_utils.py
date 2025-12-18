
import math
import os
import platform
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_conv_and_bn(conv, bn):
	fusedconv = nn.Conv2d(conv.in_channels,
						  conv.out_channels,
						  kernel_size=conv.kernel_size,
						  stride=conv.stride,
						  padding=conv.padding,
						  groups=conv.groups,
						  bias=True).requires_grad_(False).to(conv.weight.device)
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
	fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
	b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
	return fusedconv

def initialize_weights(model):
	for m in model.modules():
		t = type(m)
		if t is nn.Conv2d:
			pass
		elif t is nn.BatchNorm2d:
			m.eps = 1e-3
			m.momentum = 0.03
		elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
			m.inplace = True

def model_info(model, verbose=False, img_size=640):
	n_p = sum(x.numel() for x in model.parameters())
	n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
	if verbose:
		print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
		for i, (name, p) in enumerate(model.named_parameters()):
			name = name.replace('module_list.', '')
			print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
				  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
	print(f'Model summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients')

def profile(input, ops, n=10, device=None):
	results = []
	if not isinstance(device, torch.device):
		device = select_device(device)
	for x in input if isinstance(input, list) else [input]:
		x = x.to(device)
		x.requires_grad = True
		for m in ops if isinstance(ops, list) else [ops]:
			m = m.to(device) if hasattr(m, 'to') else m
			m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
			tf, tb, t = 0, 0, [0, 0, 0]
			try:
				for _ in range(n):
					t[0] = time_sync()
					y = m(x)
					t[1] = time_sync()
					try:
						_ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
						t[2] = time_sync()
					except Exception:
						t[2] = float('nan')
					tf += (t[1] - t[0]) * 1000 / n
					tb += (t[2] - t[1]) * 1000 / n
				mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
				s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))
				p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0
				results.append([p, mem, tf, tb, s_in, s_out])
			except Exception as e:
				print(e)
				results.append(None)
			torch.cuda.empty_cache()
	return results

def scale_img(img, ratio=1.0, same_shape=False, gs=32):
	if ratio == 1.0:
		return img
	h, w = img.shape[2:]
	s = (int(h * ratio), int(w * ratio))
	img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
	if not same_shape:
		h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
	return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

def select_device(device='', batch_size=0, newline=True):
	device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
	cpu = device == 'cpu'
	if cpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	elif device:
		os.environ['CUDA_VISIBLE_DEVICES'] = device
		assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
	cuda = not cpu and torch.cuda.is_available()
	return torch.device('cuda:0' if cuda else 'cpu')

def time_sync():
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	return time.time()
