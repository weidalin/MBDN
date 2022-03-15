import os
import time
import torch
import json
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from sys import exit

ticks = time.time()

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0]
config.onegpu = 1
# config.size_train = (1024, 2048)
# config.size_train = (224, 448)
config.size_train = (640, 1280)
config.size_test = (1024, 2048)
config.init_lr = 2e-4
config.num_epochs = 150
config.val = False
config.offset = True
config.teacher = True
# dataset
# traintransform = Compose(ToTensor())
traintransform = Compose(
    [ColorJitter(brightness=1), ToTensor()])
# traintransform = Compose(
#     [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindataset = CityPersons(path=config.train_path, type='train', config=config,
                           transform=traintransform)
trainloader = DataLoader(traindataset, batch_size=config.onegpu * len(config.gpu_ids))

# net
print('Net...')
net = ACSPNet().cuda()

version = 'V0_%s_1centergaussmap_originaladdsenetinresnet_%d_%d_%dgpuper%dimg_lr%s' % (
net.resnetType, config.size_train[0], config.size_train[1], len(config.gpu_ids), config.onegpu, config.init_lr)
# To continue training
resume_from = ''
# resume_from = './models/%s/ckpt/ACSP_12.pth.tea'%version
begin_epoch_num = 0
if len(resume_from) > 0:
    net.load_state_dict(torch.load(resume_from))
    begin_epoch_num = resume_from[resume_from.rfind('_') + 1:resume_from.rfind('.pth')]
print("begin_epoch_num:", begin_epoch_num)

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

# optimizer
params = []
for n, p in net.named_parameters():
    if p.requires_grad:
        params.append({'params': p})
    else:
        print(n)

if config.teacher:
    teacher_dict = net.state_dict()

# net = nn.DataParallel(net, device_ids=config.gpu_ids)
net = nn.DataParallel(net, device_ids=config.gpu_ids)

optimizer = optim.Adam(params, lr=config.init_lr)

batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(trainloader)

config.print_conf()

def train():


    # open log file
    log_floder = './models/' + version + '/log/'
    log_file = log_floder + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.log'
    log = open(log_file, 'w')


    for epoch in range(int(begin_epoch_num), 150):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))

        net.train()

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # add visulize by weida 2020-11-14------------------------------------------------------------------------------
            pos, height, offset = labels
            import cv2
            vis_cla_pos = pos[0, :, 0].cpu().detach().numpy()  # (5, 256, 512)
            original_img = inputs[0].cpu().numpy().transpose(1, 2, 0)  # (1024, 2048, 3)

            vis_cla_pos = vis_cla_pos.transpose(1, 2, 0)  # (256, 512, 5)
            vis_cla_pos = cv2.resize(vis_cla_pos, (1280, 640))  # (1024, 2048, 5)
            vis_cla_pos = vis_cla_pos.transpose(2, 0, 1)

            loc_type = ["top", "center", "bottom", "visible", "all"]
            for j in range(vis_cla_pos.shape[0]):
                gray_img = vis_cla_pos[j]
                heatmap = np.uint8(255 * gray_img)  # 将热力图转换为RGB格式
                heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                original_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                merge = cv2.addWeighted(original_img, 0.7, heatmap_img, 0.6, 0, dtype=cv2.CV_32F)
                basename = "/home/weida/datasets/cityperson/leftImg8bit_trainvaltest/leftImg8bit/train_vishm/" + \
                           loc_type[
                               j] + "/" + loc_type[j] + "_" + str(i) + ".png"
                cv2.imwrite(basename, merge)
            gray_img = vis_cla_pos.sum(axis=0)
            heatmap = np.uint8(255 * gray_img)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            original_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            merge = cv2.addWeighted(original_img, 0.7, heatmap_img, 0.6, 0, dtype=cv2.CV_8U)
            # cv2.imshow("original_img", original_img)
            # cv2.waitKey(0)
            # cv2.destroyWindow()
            basename = "/home/weida/datasets/cityperson/leftImg8bit_trainvaltest/leftImg8bit/train_vishm/" + loc_type[
                -1] + "/" + loc_type[-1] + "_" + str(i) + ".png"
            cv2.imwrite(basename, merge)
            print("end %d visualize" % i)
    log.close()


if __name__ == '__main__':
    train()
