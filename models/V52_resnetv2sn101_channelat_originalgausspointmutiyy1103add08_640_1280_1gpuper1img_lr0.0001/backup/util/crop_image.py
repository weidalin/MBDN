from PIL import Image
# pil paste可以进行图片拼接
import cv2
import numpy as np
import glob as glob
import os
import matplotlib.pyplot as plt
# -*-coding: UTF-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt




def file_name(root_path,picturetype):
     filename=[]
     for root,dirs,files in os.walk(root_path):
         for file in files:
             if os.path.splitext(file)[1]==picturetype:
                 filename.append(os.path.join(root,file))
     return filename

def crop_one_picture(img_path, y1,y2,x1,x2):
    image = cv2.imread(img_path)
    cropImg = image[int(y1):int(y2), int(x1):int(x2)]
    print(img_path[:-5]+'crop.jpg')
    cv2.imwrite(img_path[:-5]+'_crop.jpg', cropImg)

train_jpgs = np.array(glob.glob('/mnt/D0D8D177D8D15C72/vis/1/1661_*.png'))

for each_img in train_jpgs:
    x1, y1 ,x2 = 540, 160,  1200
    x1, y1, w, h = x1, y1, x2-y1, (x2-y1)*0.8
    img = Image.open(each_img)
    img = np.array(img)  # 获得numpy对象, np.ndarray, RGB
    plt.imshow(img)
    # plt.show()
    crop_one_picture(each_img, y1, y1+h, x1, x1+w)