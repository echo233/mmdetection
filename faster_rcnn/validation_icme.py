import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO

import numpy as np
random_num = 5
# read test result
test_path = "/liu/code/d2l-zh/pycharm-local/faster_rcnn/record_icme_2.json"
with open(test_path,'r') as f:
    test_result = json.load(f)

# read test result
test_img_ann = "/liu/icme/dataset/val/annotations/instances_val2017.json"
with open(test_img_ann,'r') as f:
    test_img_ann = json.load(f)

# read class_indict
label_json_path = './coco91_indices.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as f:
    class_dict = json.load(f)
coco_classes = class_dict



# 解决中文字体乱码
font1 = ImageFont.truetype('/usr/share/fonts/truetype/arphic/ukai.ttc', 24)

for item in random.sample(test_result,random_num):
    ids = []
    for i,tmp in enumerate(test_result):
        if tmp["image_id"] == item["image_id"]:
            ids.append(i)

    img = os.path.join("/liu/icme/dataset/val/images", test_img_ann["images"][item["image_id"]]["file_name"])
    original_img = Image.open(img)
    id = str(item["category_id"])

    if len(item["bbox"]) == 0:
        print("没有检测到任何目标!")

    draw = ImageDraw.Draw(original_img)
    for j in range(len(ids)):
        item = test_result[ids[j]]
        # draw box to image
        x, y, w, h = item["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2),outline='red')
        draw.text((x1, y1),coco_classes[id],fill='red',font=font1)

    # show image
    plt.imshow(original_img)
    pylab.show()




