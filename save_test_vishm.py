import os
import time
import torch
import json
import numpy as np
import time
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from util.functions import parse_det_offset
from eval_city.eval_script.eval_demo import validate
from sys import exit
from util import draw_bbox

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = Config()
config.test_path = './data/citypersons'
config.size_test = (1280, 2560)
config.init_lr = 2e-4
config.offset = True
config.val = True
config.val_frequency = 1
config.teacher = True
config.print_conf()

# dataset
testtransform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
testdataset = CityPersons(path=config.train_path, type='val', config=config, transform=testtransform, preloaded=True)
testloader = DataLoader(testdataset, batch_size=1)

# net
print('Net...')
net = ACSPNet().cuda()

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

teacher_dict = net.state_dict()


def val(r, name, log=None):
    base_path = "/mnt/D0D8D177D8D15C72/cityperson_visualize_heatmap/%s/"%(name[name.find('V') : name.find('_')])
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    net.eval()
    # load the model here!!!
    teacher_dict = torch.load(name)
    net.load_state_dict(teacher_dict)

    # print(net)
    print('Perform validation...')
    res = []
    t3 = time.time()
    for i, data in enumerate(testloader, 0):
        inputs = data.cuda()  # torch.Size([1, 3, 1024, 2048])

        with torch.no_grad():
            pos, height, offset = net(inputs)
        # add visulize by weida 2020-11-14------------------------------------------------------------------------------
        #plot heatmap
        import cv2
        vis_cla_pos = pos[0].permute(1,2,0).cpu().numpy() #torch.Size([256, 512, 2])
        vis_cla_pos = cv2.resize(vis_cla_pos, (2048, 1024))  # (1024, 2048, 5)
        vis_cla_pos = vis_cla_pos.transpose(2, 0, 1)
        original_img = data[0].permute(1, 2, 0).cpu().numpy()  # torch.Size([1024, 2048, 3])
        original_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        loc_type = ["top", "full", "all"]
        for j in range(vis_cla_pos.shape[0]):
            gray_img = vis_cla_pos[j]

            # imgH, imgW = gray_img.shape
            # mean_gray_img = np.mean(gray_img)
            # crop_mask = gray_img > mean_gray_img
            # nonzero_indices = torch.nonzero(crop_mask)
            # padding_ratio = 0.1
            # height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            # height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            # width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            # width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            # crop = gray_img[height_min:height_max, width_min:width_max]
            # cv2.imshow("crop", crop)
            #

            # gray_img = (gray_img - np.argmin(gray_img))/ (np.argmax(gray_img) - np.argmin(gray_img))
            heatmap = np.uint8(255 * gray_img)  # 将热力图转换为RGB格式
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            merge = cv2.addWeighted(original_img, 0.4, heatmap_img, 0.7, 0)

            basename = base_path + str(i) + "_" + loc_type[j] + ".png"
            print(basename)
            cv2.imwrite(basename, merge)
        gray_img = vis_cla_pos.sum(axis=0)
        heatmap = np.uint8(255 * gray_img)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        merge_all = cv2.addWeighted(original_img, 0.4, heatmap_img, 0.7, 0)
        basename = base_path + str(i) + "_" + loc_type[-1] + ".png"
        print(basename)
        cv2.imwrite(basename, merge_all)

        # plot boxes----------------------------------
        dtboxes = parse_det_offset(r, pos.cpu().numpy(), height.cpu().numpy(), offset.cpu().numpy(), config.size_test,
                                 score=0.3, down=4, nms_thresh=0.5)

        box_hm_img = draw_bbox.draw_boxes_bbox_with_hm(merge_all, dtboxes, line_thick=2, line_color='blue')
        box_ori_img = draw_bbox.draw_boxes_bbox_with_hm(original_img, dtboxes, line_thick=2, line_color='blue')
        hm_basename = base_path + str(i) + "_hm_bbox" + ".png"
        ori_basename = base_path + str(i) + "_ori_bbox" + ".png"
        cv2.imwrite(hm_basename, box_hm_img)
        cv2.imwrite(ori_basename, box_ori_img)
        print("end %d visualize" % i)
        # end add visulize ------------------------------------------------------------------------------


# or Val your own model
version = 'V20_resnetv2sn50_headandfull2brand2center2gaussmap_640_1280_1gpuper1img_lr0.0002'
log_floder = './models/' + version + '/validation_result_log/'
log_file = log_floder + version + time.strftime('val_log_%Y%m%d_%H%M%S', time.localtime(time.time())) + '.log'
if not os.path.exists(log_floder):
    os.mkdir(log_floder)
log = open(log_file, 'w')
for i in range(9, 150):
    name = './models/' + version + '/ckpt/ACSP_{0}.pth.tea'.format(i)
    if not os.path.exists(name):
        continue;
    val(0.36, name, log)

# name_1 = './models/ACSP(Smooth L1).pth.tea'
# name_2 = './models/ACSP(Vanilla L1).pth.tea'
# val(0.40, name_2)
# val(0.36, name_2)
