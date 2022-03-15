import os
os.path.join("../")
import cv2
import numpy as np
from eval_city.eval_script.coco import COCO
from config import Config

config = Config()
config.test_path = '../data/citypersons'


def get_anno(annFile):
    # for id_setup in range(0, 4):
    cocoGt = COCO(annFile)
    return cocoGt
color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}


def gaussian(kernel):
    sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))

def draw_boxes(img, boxes, category_ids, cats, scores=None, tags=None, line_thick=1, line_color='white', vis=False):

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        w, h = x2 - x1, y2 - y1
        if vis == True:
            print("plotting green circle")
            cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), int(np.log(h)), color['orange'], -1)
            # cv2.rectangle(img, (x1,y1), (x2,y2), color[line_color], line_thick)
            # continue
        else:
            cv2.rectangle(img, (x1,y1), (x2,y2), color[line_color], line_thick)
            #3center
            xx11, yy11, xx12, yy12 = int(x1), int(y1), int(x2), int(y1 + 0.2 * h)
            c_x1, c_y1 = int((xx11 + xx12) / 2), int((yy11 + yy12) / 2)
            xx21, yy21, xx22, yy22 = int(x1), int(y1 + 0.2 * h), int(x2), int(y1 + 0.8 * h)
            c_x2, c_y2 = int((xx21 + xx22) / 2), int((yy21 + yy22) / 2)
            xx31, yy31, xx32, yy32 = int(x1 + 0.2 * w), int(y1 + 0.8 * h), int(x2), int(y2)
            c_x3, c_y3 = int((xx31 + xx32) / 2), int((yy31 + yy32) / 2)

            #end 3center
            cv2.circle(img, (c_x1, c_y1), int(np.log(h)), color['red'], 0)
            cv2.circle(img, (c_x2, c_y2), int(np.log(h)), color['red'], 0)
            cv2.circle(img, (c_x3, c_y3), int(np.log(h)), color['red'], 0)
            #1center
            # cv2.circle(img, ((x2-x1)/2, (y2-y1)/2), int(np.log(h)), color['red'], 0)

        if category_ids is not None and not vis:
            # self.HtRng = [[50, 1e5 ** 2], [50,75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
            # self.VisRng = [[0.65, 1e5 ** 2], [0.65, 1e5 ** 2], [0.2,0.65], [0.2, 1e5 ** 2]]
            # self.SetupLbl = ['Reasonable', 'Reasonable_small','Reasonable_occ=heavy', 'All']
            if heights[i]>=50 and vis_ratios[i]>=0.65:
                tag = "Reason"
            elif heights[i]>=50 and heights[i] <= 75 and vis_ratios[i]>0.65:
                tag = "Rea_small"
            elif heights[i]>=20 and vis_ratios[i]>=0.2 and vis_ratios[i] <= 0.65:
                tag = "Heavy"
            elif heights[i]>=20 and vis_ratios[i]>=0.2:
                tag = "All"
            else:
                tag = "Else"

            text = "{}".format(tag)
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color[line_color], line_thick)
    return img

anno = get_anno('../eval_city/val_gt.json')
anns_id = 1
img_paths = anno.imgs
anns = anno.anns
cats = anno.cats
for i in range(1,501):
    # img_path = anno['imgs'][i]['im_name']
    img_path = img_paths[i]['im_name']
    print("imwrite %d image: %s"%(i, img_path))
    imgage_id = img_paths[i]['id']
    image = cv2.imread("./data/citypersons/images/val/%s"%img_path)
    bboxes = []
    vboxes = []
    heights = []
    category_ids = []
    vis_ratios = []

    while anns[anns_id]['image_id']==imgage_id:
        imgage_id = anns[anns_id]['image_id']
        bbox = anns[anns_id]['bbox']
        ignore = anns[anns_id]["ignore"]
        if ignore == 1:
            anns_id +=1
            continue

        bbox[2] = bbox[2]+bbox[0]
        bbox[3] = bbox[3]+bbox[1]
        vbox = anns[anns_id]['vis_bbox']
        vbox[2] = vbox[2]+vbox[0]
        vbox[3] = vbox[3]+vbox[1]
        vis_ratio = anns[anns_id]['vis_ratio']
        category_id = anns[anns_id]["category_id"]
        category_ids.append(category_id)
        anns_id+=1
        height = anns[anns_id]['height']
        heights.append(height)
        vis_ratios.append(vis_ratio)
        bboxes.append(bbox)
        vboxes.append(vbox)
    if image is None:
        print("can't find %s"%img_path)
    image = draw_boxes(image, vboxes, category_ids, cats, heights, vis_ratios, line_thick=2, line_color='blue', vis=True)
    image = draw_boxes(image, bboxes, category_ids, cats, heights,  vis_ratios, line_thick=2, line_color='white', vis=False)
    img_save_path = '/home/weida/datasets/cityperson/leftImg8bit_trainvaltest/leftImg8bit/vis_val/%s'%img_path
    cv2.imwrite(img_save_path , image)

