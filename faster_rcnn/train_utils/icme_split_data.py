from pycocotools.coco import COCO
import os
import random
import json
import shutil

root = "/liu/icme/dataset/train"
#============================划分train/val代码==========================================================
image_root = "/liu/icme/dataset/train/images"
coco = COCO('/liu/icme/dataset/train/annotations/instances_train2017.json')
class_names = [coco.cats[catId]['name'] for catId in coco.getCatIds()]
categories = [dict(id=i + 1, name=name) for i, name in enumerate(class_names)]

annotaions_train = []
images_train = []
annotaions_val = []
images_val = []

images_trainsss = []
for catId in coco.getCatIds():
    imgIds = coco.getImgIds(catIds=[catId])
    random.shuffle(imgIds)
    for imgId in imgIds[:10]:
        img = coco.imgs[imgId]
        images_val.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_val.append(ann)
    for imgId in imgIds[10:]:
        img = coco.imgs[imgId]
        images_train.append(img)
        images_trainsss.append(img["file_name"])
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            annotaions_train.append(ann)
train_annFile = {"images": images_train, "type": "instances", "annotations": annotaions_train,
                 "categories": categories}
val_annFile = {"images": images_val, "type": "instances", "annotations": annotaions_val, "categories": categories}


path2 = "/liu/icme/dataset/train/images"  # 文件夹目录
files = os.listdir(path2)  # 得到文件夹下的所有文件名称
val_images = "/liu/icme/dataset/train/val_images/"
for file in files:  # 遍历文件夹
    if file not in images_trainsss:
        shutil.move(os.path.join(image_root,file), val_images)
        #os.remove(os.path.join(path2,file))

instances_train = '/liu/icme/dataset/train/annotations/instances_train.json'
instances_val = '/liu/icme/dataset/train/annotations/instances_val.json'
json_str1 = json.dumps(train_annFile, ensure_ascii=False, indent=4)  # 缩进4字符
json_str2 = json.dumps(val_annFile, ensure_ascii=False, indent=4)  # 缩进4字符
with open(instances_train, 'w') as json_file1:
    json_file1.write(json_str1)
with open(instances_val, 'w') as json_file2:
    json_file2.write(json_str2)
json_file1.close()
json_file2.close()

# ===================================================================================

# ===================================================================================
