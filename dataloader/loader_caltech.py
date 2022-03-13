from __future__ import division
import sys
import random
import cv2
import util
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import dataloader.data_augment
from dataloader.load_data import get_citypersons, get_caltech


class CityPersons(Dataset):
    def __init__(self, path, type, config, preloaded=False, transform=None, caffemodel=False):

        # self.dataset = get_citypersons(root_dir=path, type=type)
        self.dataset = get_caltech("/mnt/D0D8D177D8D15C72/datasets/caltech_cityperson", type)
        # self.dataset = self.dataset[:50]
        self.dataset_len = len(self.dataset)
        self.type = type

        if self.type == 'train' and config.train_random:
            random.shuffle(self.dataset)
        self.config = config
        self.transform = transform
        self.caffemodel = caffemodel

        if self.type == 'train':
            self.preprocess = RandomResizeFix(size=config.size_train, scale=(0.4, 1.5))
        else:
            self.preprocess = None#ResizeFix(size=config.size_test, scale=1.25)#wwh

        self.preloaded = preloaded

        if self.preloaded:
            self.img_cache = []
            where = []
            for i, data in enumerate(self.dataset):
                if self.caffemodel:
                    self.img_cache.append(cv2.imread(data['filepath']))
                else:
                    img = Image.open(data['filepath'])
                    #out = img.resize(tuple(map(lambda x: int(x * 1.25), img.size)))
                    self.img_cache.append(img)
                where.append(data['filepath'])
                np.save("order.npy",np.array(where))
                # print('%d/%d\r' % (i+1, self.dataset_len)),
                # sys.stdout.flush()
            print('')
       

    def __getitem__(self, item):
        if self.caffemodel:
            # input is BGR order, not normalized
            img_data = self.dataset[item]
            if self.preloaded:
                img = self.img_cache[item]
            else:
                img = cv2.imread(img_data['filepath'])

            if self.type == 'train':
                img_data, x_img = data_augment.augment(self.dataset[item], self.config, img)

                gts = img_data['bboxes'].copy()
                igs = img_data['ignoreareas'].copy()

                y_center, y_height, y_offset = self.calc_gt_center(gts, igs, radius=2, stride=self.config.down)

                x_img = x_img.astype(np.float32)
                x_img -= [103.939, 116.779, 123.68]
                x_img = torch.from_numpy(x_img).permute([2, 0, 1])

                return x_img, [y_center, y_height, y_offset]

            else:
                x_img = img.astype(np.float32)
                x_img -= [103.939, 116.779, 123.68]
                x_img = torch.from_numpy(x_img).permute([2, 0, 1])

                return x_img
        else:
            # input is RGB order, and normalized
            img_data = self.dataset[item]
            if self.preloaded:
                img = self.img_cache[item]
            else:
                img = Image.open(img_data['filepath'])
                
            if self.type == 'train':
                gts = img_data['bboxes'].copy()
                igs = img_data['ignoreareas'].copy()
                v_gts = img_data['vis_bboxes'].copy()
                x_img, v_gts, gts, igs = self.preprocess(img, v_gts, gts, igs)
                #(3, 96, 192) (2, 96, 192) (3, 96, 192)
                y_center, y_height, y_offset = self.calc_gt_center(v_gts, gts, igs, radius=2, stride=self.config.down)
                if self.transform is not None:
                    x_img = self.transform(x_img)#torch.Size([3, 384, 768])


                return x_img, [y_center, y_height, y_offset]

            else:
                
                if self.transform is not None:
                    x_img = self.transform(img)
                else:
                    x_img = img
                
                return x_img

    def __len__(self):
        return self.dataset_len

    def calc_gt_center(self, v_gts, gts, igs, radius=2, stride=4):

        def gaussian(kernel):
            sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
            s = 2*(sigma**2)
            dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
            return np.reshape(dx, (-1, 1))

        scale_map = np.zeros((6, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        offset_map = np.zeros((9, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride)))
        pos_map = np.zeros((9, int(self.config.size_train[0] / stride), int(self.config.size_train[1] / stride))) #modify by weida 2020-10-21 3-->5
        #012->gaussmap 3->mask 456->center
        pos_map[3, :, :, ] = 1  # channel 1: 1-value mask, ignore area will be set to 0

        if len(igs) > 0:
            igs = igs / stride
            for ind in range(len(igs)):
                x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))#ceil(x2y2)
                pos_map[1, y1:y2, x1:x2] = 0    #CHW

        if len(gts) > 0:
            gts = gts / stride
            v_gts = v_gts / stride
            for ind in range(len(gts)):
                # add vbox by weida 2020-11-02
                v_x1, v_y1, v_x2, v_y2 = int(np.ceil(v_gts[ind, 0])), int(np.ceil(v_gts[ind, 1])), int(
                    v_gts[ind, 2]), int(v_gts[ind, 3])  # ceil(x1y1)
                v_c_x, v_c_y = int((v_gts[ind, 0] + v_gts[ind, 2] - 1) / 2), int(
                    (v_gts[ind, 1] + v_gts[ind, 3] - 1) / 2)  # got center coordinate
                v_dx = gaussian(v_x2 - v_x1)
                v_dy = gaussian(v_y2 - v_y1)
                v_gau_map = np.multiply(v_dy, np.transpose(v_dx))  # h*w
                # end add vbox

                x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])#ceil(x1y1)
                # c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)#got center coordinate
                print(x1, y1, x2, y2, "----", v_x1, v_y1, v_x2, v_y2)
                w, h = x2-x1, y2-y1
                xx11, yy11, xx12, yy12 = int(x1), int(y1), int(x2), int(y1+0.2*h)
                c_x1, c_y1 = int((xx11+xx12)/2), int((yy11+yy12)/2)
                xx21, yy21, xx22, yy22 = int(x1), int(y1+0.2*h), int(x2), int(y1+0.8*h)
                c_x2, c_y2 = int((xx21+xx22)/2), int((yy21+yy22)/2)
                xx31, yy31, xx32, yy32 = int(x1), int(y1+0.8*h), int(x2), int(y2)
                c_x3, c_y3 = int((xx31+xx32)/2), int((yy31+yy32)/2)

                dxx1 = gaussian(xx12-xx11) #dx = gaussian(w)
                dyy1 = gaussian(yy12-yy11) #dy = gaussian(h)
                gau_map1 = np.multiply(dyy1, np.transpose(dxx1)) #h*w
                dxx2 = gaussian(xx22-xx21) #dx = gaussian(w)
                dyy2 = gaussian(yy22-yy21) #dy = gaussian(h)
                gau_map2 = np.multiply(dyy2, np.transpose(dxx2)) #h*w
                dxx3 = gaussian(xx32-xx31) #dx = gaussian(w)
                dyy3 = gaussian(yy32-yy31) #dy = gaussian(h)
                gau_map3 = np.multiply(dyy3, np.transpose(dxx3)) #h*w

                pos_map[0, yy11:yy12, xx11:xx12] = np.maximum(pos_map[0, yy11:yy12, xx11:xx12], gau_map1)  # gauss map
                pos_map[1, yy21:yy22, xx21:xx22] = np.maximum(pos_map[1, yy21:yy22, xx21:xx22], gau_map2)  # gauss map
                pos_map[2, yy31:yy32, xx31:xx32] = np.maximum(pos_map[2, yy31:yy32, xx31:xx32], gau_map3)  # gauss map
                pos_map[3, y1:y2, x1:x2] = 1  # 1-mask map
                pos_map[4, c_y1, c_x1] = 1  # center point = 1
                pos_map[5, c_y2, c_x2] = 1  # center point = 1
                pos_map[6, c_y3, c_x3] = 1  # center point = 1
                pos_map[7, v_y1:v_y2, v_x1:v_x2] = np.maximum(pos_map[0, v_y1:v_y2, v_x1:v_x2], v_gau_map)  # gauss map
                pos_map[8, v_c_y, v_c_x] = 1  # gauss map

                #end modify---------------------------------------------------------------------------------------------------

                scale_map[0, c_y1-radius:c_y1+radius+1, c_x1-radius:c_x1+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[1, c_y2-radius:c_y2+radius+1, c_x2-radius:c_x2+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[2, c_y3-radius:c_y3+radius+1, c_x3-radius:c_x3+radius+1] = np.log(gts[ind, 3] - gts[ind, 1])  # log value of height
                scale_map[3, c_y1-radius:c_y1+radius+1, c_x1-radius:c_x1+radius+1] = 1  # 1-mask
                scale_map[4, c_y2-radius:c_y2+radius+1, c_x2-radius:c_x2+radius+1] = 1  # 1-mask
                scale_map[5, c_y3-radius:c_y3+radius+1, c_x3-radius:c_x3+radius+1] = 1  # 1-mask

                offset_map[0, c_y1, c_x1] = (yy11+yy12) / 2 - c_y1 - 0.5  # height-Y offset
                offset_map[1, c_y1, c_x1] = (xx11+xx12) / 2 - c_x1 - 0.5  # width-X offset
                offset_map[2, c_y2, c_x2] = (yy21+yy22) / 2 - c_y2 - 0.5  # height-Y offset
                offset_map[3, c_y2, c_x2] = (xx21+xx22) / 2 - c_x2 - 0.5  # width-X offset
                offset_map[4, c_y3, c_x3] = (yy31+yy32) / 2 - c_y3 - 0.5  # height-Y offset
                offset_map[5, c_y3, c_x3] = (xx31+xx32) / 2 - c_x3 - 0.5  # width-X offset
                offset_map[6, c_y1, c_x1] = 1  # 1-mask
                offset_map[7, c_y2, c_x2] = 1  # 1-mask
                offset_map[8, c_y3, c_x3] = 1  # 1-mask

        return pos_map, scale_map, offset_map#(7, 160, 320) (6, 160, 320) (3, 160, 320)

    
    
    
    
    
    

class RandomResizeFix(object):
    """
    Args:
        size: expected output size of each edge
        scale: scale factor
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.4, 1.5), interpolation=Image.BILINEAR):
        self.size = size#(640, 1280)
        self.interpolation = interpolation
        self.scale = scale  # (0.4, 1.5)

    def __call__(self, img, v_gts, gts, igs):
        # resize image
        w, h = img.size#(2048, 1024)
        ratio = np.random.uniform(self.scale[0], self.scale[1])
        # print("ratio:", ratio)
        n_w, n_h = int(ratio * w), int(ratio * h)
        img = img.resize((n_w, n_h), self.interpolation)#随机缩放
        gts = gts.copy()
        v_gts = v_gts.copy()
        igs = igs.copy()

        # resize label
        if len(gts) > 0:
            gts = np.asarray(gts, dtype=float)
            v_gts = np.asarray(v_gts, dtype=float)
            gts *= ratio
            v_gts *= ratio

        if len(igs) > 0:
            igs = np.asarray(igs, dtype=float)
            igs *= ratio

        # random flip
        w, h = img.size
        if np.random.randint(0, 2) == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(gts) > 0:
                gts[:, [0, 2]] = w - gts[:, [2, 0]]
                v_gts[:, [0, 2]] = w - v_gts[:, [2, 0]]
            if len(igs) > 0:
                igs[:, [0, 2]] = w - igs[:, [2, 0]]  # x1yx2y --> [w-x2]y[w-x1]y????filp

        if h >= self.size[0]:
            # random crop after resize
            img, v_gts, gts, igs = self.random_crop(img, v_gts, gts, igs, self.size, limit=16)
        else:
            # random pad
            img, v_gts, gts, igs = self.random_pave(img, v_gts, gts, igs, self.size, limit=16)

        return img, v_gts, gts, igs

    @staticmethod
    def random_crop(img, v_gts, gts, igs, size, limit=8):
        # print("into random_crop......")
        w, h = img.size
        crop_h, crop_w = size #size = config.size_train

        if len(gts) > 0:#随机找一个GT框，并以这个框的中心作为crop的中心
            sel_id = np.random.randint(0, len(gts))
            sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
            sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
            v_sel_center_x = int((v_gts[sel_id, 0] + v_gts[sel_id, 2]) / 2.0)
            v_sel_center_y = int((v_gts[sel_id, 1] + v_gts[sel_id, 3]) / 2.0)
        else:#否则就随便了
            sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0)) #x1
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0)) #y1
        diff_x = max(crop_x1 + crop_w - w, int(0))#检查有没有超过边界
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))

        # crop detections
        if len(igs) > 0:
            igs[:, 0:4:2] -= crop_x1
            igs[:, 1:4:2] -= crop_y1
            igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
            igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            before_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
            gts[:, 0:4:2] -= crop_x1
            gts[:, 1:4:2] -= crop_y1
            gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
            gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)
            after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            gts = gts[keep_inds]

            v_before_area = (v_gts[:, 2] - v_gts[:, 0]) * (v_gts[:, 3] - v_gts[:, 1])
            v_gts[:, 0:4:2] -= crop_x1
            v_gts[:, 1:4:2] -= crop_y1
            v_gts[:, 0:4:2] = np.clip(v_gts[:, 0:4:2], 0, crop_w)

            v_gts[:, 1:4:2] = np.clip(v_gts[:, 1:4:2], 0, crop_h)

            v_after_area = (v_gts[:, 2] - v_gts[:, 0]) * (v_gts[:, 3] - v_gts[:, 1])
            #
            v_keep_inds = ((v_gts[:, 2] - v_gts[:, 0]) >= limit) & (v_after_area >= 0.5 * v_before_area)
            #将不符合的东西丢掉～怎么丢呢？又要保持与gts一致性----

            # v_gts[i] = np.zeros(4) for i, v_keep in enumerate(v_keep_inds) if v_keep == False else continue

            for i, v_keep in enumerate(v_keep_inds):
                if v_keep == True:
                    continue
                else:
                    print("v_gt[",i,"] set to 0",)
                    v_gts[i] = np.zeros(4)
            v_gts = v_gts[keep_inds]
            #end ---------------------

        return cropped_img, v_gts, gts, igs

    @staticmethod
    def random_pave(img, v_gts, gts, igs, size, limit=8):
        img = np.asarray(img)
        h, w = img.shape[0:2]#after resize:(1101, 550)
        pave_h, pave_w = size #(640, 1280)
        # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
        paved_image = np.ones((pave_h, pave_w, 3), dtype=img.dtype) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img
        import cv2
        # cv2.imshow("paved_image", paved_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pave detections
        if len(igs) > 0:
            igs[:, 0:4:2] += pave_x
            igs[:, 1:4:2] += pave_y
            keep_inds = ((igs[:, 2] - igs[:, 0]) >= 8) & ((igs[:, 3] - igs[:, 1]) >= 8)
            igs = igs[keep_inds]

        if len(gts) > 0:
            gts[:, 0:4:2] += pave_x
            v_gts[:, 0:4:2] += pave_x
            gts[:, 1:4:2] += pave_y
            v_gts[:, 1:4:2] += pave_y
            keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
            v_keep_inds = ((v_gts[:, 2] - v_gts[:, 0]) >= limit)
            gts = gts[keep_inds]
            v_gts = v_gts[keep_inds]

        return Image.fromarray(paved_image), v_gts, gts, igs


