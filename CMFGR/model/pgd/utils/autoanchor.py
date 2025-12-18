
def check_anchor_order(m):
	# Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
	a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
	da = a[-1] - a[0]  # delta a
	ds = m.stride[-1] - m.stride[0]  # delta s
	if da and (da.sign() != ds.sign()):  # same order
		# 这里LOGGER可用print替代，或自行实现
		print('AutoAnchor: Reversing anchor order')
		m.anchors[:] = m.anchors.flip(0)
