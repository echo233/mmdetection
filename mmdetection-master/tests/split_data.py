import os
from pycocotools.coco import COCO
import random
import json

root = "/liu/icme/dataset/train/annotations/"

coco = COCO(os.path.join(root,'instances_train2017.json'))
class_names = [coco.cats[catId]['name'] for catId in coco.getCatIds()]
categories = [dict(id=i+1, name=cat) for i, cat in enumerate(class_names)]

train_images = []
tarain_annotation = []
val_images = []
val_annotation = []
for catId in coco.getCatIds():
    imgIds = coco.getImgIds(catIds=[catId])
    random.shuffle(imgIds)
    for imgId in imgIds[:10]:
        img = coco.imgs[imgId]
        val_images.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            val_annotation.append(ann)
    for imgId in imgIds[10:]:
        img = coco.imgs[imgId]
        train_images.append(img)
        anns = coco.imgToAnns[imgId]
        for ann in anns:
            tarain_annotation.append(ann)

json_dict_train = {"images": train_images, "type": "instances", "annotations": tarain_annotation, "categories": categories}
json_dict_val = {"images": val_images, "type": "instances", "annotations": val_annotation, "categories": categories}

train_annotation = root + "/train_annotation.json"
val_annotation = root + "/val_annotation.json"

train_json = json.dumps(json_dict_train, indent=4, ensure_ascii=False)
with open(train_annotation, 'w', encoding='utf-8') as f1:
    f1.write(train_json)

val_json = json.dumps(json_dict_val, indent=4, ensure_ascii=False)
with open(val_annotation, 'w', encoding='utf-8') as f2:
    f2.write(val_json)

print(1)
