import torch
import tensorwatch as tw

# 其实就两句话
from net.network_sn_101 import ACSPNet

net = ACSPNet().cuda()
#2
# outputs = net(inputs)
# g = make_dot(outputs)
# g.render('resnet50_model', view=False)
#3.
tw.draw_model(net, [1, 3, 512, 256])