from __future__ import division
import os
import numpy as np
from scipy import io as scio


def get_citypersons(root_dir='./data/citypersons', type='train'):
    all_img_path = os.path.join(root_dir, 'images')
    all_anno_path = os.path.join(root_dir, 'annotations')
    rows, cols = 1024, 2048

    anno_path = os.path.join(all_anno_path, 'anno_' + type + '.mat')
    res_path = os.path.join('./data/cache/cityperson', type)
    image_data = []
    annos = scio.loadmat(anno_path)
    index = 'anno_' + type + '_aligned'
    valid_count = 0
    iggt_count = 0
    box_count = 0

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        # print(anno)
        cityname = anno[0][0][0][0].encode('utf8')
        imgname = anno[0][0][1][0].encode('utf8')
        gts = anno[0][0][2]
        img_path = os.path.join(all_img_path, str(type) + '/' + str(cityname, encoding = "utf-8") + '/' + str(imgname, encoding = "utf-8"))
        boxes = []
        ig_boxes = []
        vis_boxes = []
        for i in range(len(gts)):
            label, x1, y1, w, h = gts[i, :5]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
            xv1, yv1, wv, hv = gts[i, 6:]
            xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
            wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)

            if label == 1 and h >= 50:
                box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])#xywh --> xyxy
                boxes.append(box)
                vis_box = np.array([int(xv1), int(yv1), int(xv1) + int(wv), int(yv1) + int(hv)])#xywh --> xyxy
                vis_boxes.append(vis_box)
            else:
                ig_box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                ig_boxes.append(ig_box)
        boxes = np.array(boxes)
        vis_boxes = np.array(vis_boxes)
        ig_boxes = np.array(ig_boxes)

        if len(boxes) > 0:
            valid_count += 1
        annotation = {}
        annotation['filepath'] = img_path
        box_count += len(boxes)
        iggt_count += len(ig_boxes)
        annotation['bboxes'] = boxes
        annotation['vis_bboxes'] = vis_boxes
        annotation['ignoreareas'] = ig_boxes
        image_data.append(annotation)

    return image_data



import json
import cv2
def get_caltech(inputdir='', type='train'):
    image_data = []
    anno_path = os.path.join(inputdir, type)

    rows, cols = 640, 480
    for (dirpath, dirnames, filenames) in os.walk(anno_path):
        for filename in filenames:

            valid_count = 0
            iggt_count = 0
            box_count = 0
            if os.path.splitext(filename)[1] == '.json':
                json_path = os.path.join(dirpath, filename)
                img_path = os.path.join(dirpath.replace("annotations", "images"), filename.replace(".json", ".jpg"))

                with open(json_path, 'r') as load_f:
                    load_dicts = json.load(load_f)
                    boxes = []
                    ig_boxes = []
                    vis_boxes = []

                    # image = cv2.imread(img_path)
                    for load_dict in load_dicts:
                        label = 1
                        x1, y1, w, h = load_dict["pos"]
                        # print(x1, y1, w, h)
                        # xmin = int(x1)
                        # xmax = int(x1 + w)
                        # ymin = int(y1)
                        # ymax = int(y1 + h)
                        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        x1, y1 = max(int(x1), 0), max(int(y1), 0)
                        w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
                        # print(img_path)
                        # print(load_dict["posv"])
                        if isinstance (load_dict["posv"],int):
                            xv1, yv1, wv, hv = 0, 0, 0, 0
                        else:
                            xv1, yv1, wv, hv = load_dict["posv"]
                        xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
                        wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)

                        if label == 1 and h >= 20:
                            box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])  # xywh --> xyxy
                            boxes.append(box)
                            vis_box = np.array(
                                [int(xv1), int(yv1), int(xv1) + int(wv), int(yv1) + int(hv)])  # xywh --> xyxy
                            vis_boxes.append(vis_box)
                        else:
                            ig_box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                            ig_boxes.append(ig_box)

                    # cv2.imshow("draw_1", image)
                    # cv2.waitKey(0)
                    boxes = np.array(boxes)
                    vis_boxes = np.array(vis_boxes)
                    ig_boxes = np.array(ig_boxes)

                    if len(boxes) > 0:
                        valid_count += 1
                    annotation = {}
                    annotation['filepath'] = img_path
                    box_count += len(boxes)
                    iggt_count += len(ig_boxes)
                    annotation['bboxes'] = boxes
                    annotation['vis_bboxes'] = vis_boxes
                    annotation['ignoreareas'] = ig_boxes
                    # print(img_path + "\t" + str(len(boxes)) + "\t" + str(len(vis_boxes)) + "\t" + str(len(ig_boxes)))
                    image_data.append(annotation)
    return image_data


def load_gt(dict_input, key_name, key_box, class_names):

    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        keys = key_box.split(',')
        for key_ in keys:
            assert key_ in dict_input[key_name][0]
    bbox = []
    ig_bbox = []
    for rb in dict_input[key_name]:
        if rb['tag'] == 'person':  # modify[3]: 'person' --> '1'
            for cls in class_names[1:]:  # cls='person'
                if 'extra' in rb:
                    if 'ignore' in rb['extra']:
                        if rb['extra']['ignore'] != 0:
                            tag = -1
                bbox.append((rb[cls]))
        else:  # mask
            for cls in class_names[1:]:
                ig_bbox.append((rb[cls]))

    bboxes = np.vstack(bbox).astype(np.float64)
    ig_bboxes = np.vstack(ig_bbox).astype(np.float64)
    return bboxes, ig_bboxes

def get_crowdhuman(root_dir='./data/citypersons', type='train'):

    all_img_path = os.path.join(root_dir, 'Images')
    all_anno_path = os.path.join(root_dir, 'annotation_' + type + '.odgt')
    with open(all_anno_path, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]

    rows, cols = 1024, 2048

    image_data = []
    index = 'anno_' + type + '_aligned'
    valid_count = 0
    iggt_count = 0
    box_count = 0

    for l in range(len(records)):
        record = records[l]
        boxes = []
        ig_boxes = []
        vis_boxes = []
        img_path = os.path.join(all_img_path, record["ID"] +".jpg")

        for rb in record["gtboxes"]:
            if rb['tag'] == 'person':  # modify[3]: 'person' --> '1'
                boxes.append((rb["fbox"]))
                vis_boxes.append(rb["vbox"])

        if len(boxes) > 0:
            valid_count += 1
        annotation = {}
        annotation['filepath'] = img_path
        box_count += len(boxes)
        iggt_count += len(ig_boxes)
        if len(boxes) != len(vis_boxes):
            print(len(boxes))
            print(len(vis_boxes))
        annotation['bboxes'] = boxes
        annotation['vis_bboxes'] = vis_boxes
        annotation['ignoreareas'] = ig_boxes
        image_data.append(annotation)

    return image_data

if __name__ == "__main__":
    # caltech_image_data = get_caltech("/mnt/D0D8D177D8D15C72/datasets/caltech_cityperson", "train")
    # print(len(caltech_image_data))

    # cityperson_image_data = get_citypersons('../data/citypersons', 'train')
    # print(len(cityperson_image_data))

    crowdhuman_image_data = get_crowdhuman("/mnt/D0D8D177D8D15C72/datasets/crowdhuman", "train")

    print(len(crowdhuman_image_data))



