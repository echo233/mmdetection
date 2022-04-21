import os
import json

import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO

def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids



class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, dataset="train", transforms=None, years="2017"):
        super(CocoDetection, self).__init__()

        if dataset == "train":
            self.img_root = "/liu/icme/dataset/train/images"
            self.anno_path = "/liu/icme/dataset/train/annotations/instances_train.json"
        elif dataset == "val":
            self.img_root = "/liu/icme/dataset/train/val_images"
            self.anno_path = "/liu/icme/dataset/train/annotations/instances_val.json"
        elif dataset == "test":
            self.img_root = "/liu/icme/dataset/val/images"
            self.anno_path = "/liu/icme/dataset/val/annotations/instances_val2017.json"

        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        # 获取coco数据索引与类别名称的关系
        # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90
        # 将缺失的类别名称设置成N/A
        coco_classes = {}
        for k in range(1, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("coco91_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])



        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# import transforms
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# import os
# from pycocotools.coco import COCO
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import pylab
#
# # 解决中文字体乱码
# font1 = ImageFont.truetype('/usr/share/fonts/truetype/arphic/ukai.ttc', 24)
#
# json_path = "/liu/icme/dataset/train/annotations/instances_train.json"
# img_path = "/liu/icme/dataset/val/images"
#
# # load coco data
# #coco = COCO(annotation_file=json_path)
# train_dataset = CocoDetection("/liu/icme/dataset/", dataset="test", transforms=data_transform["val"], years="")
# train_data_loader = torch.utils.data.DataLoader(train_dataset,
#                                                 batch_size=1,
#                                                 shuffle=False,
#                                                 pin_memory=True,
#                                                 num_workers=8,
#                                                 collate_fn=train_dataset.collate_fn)
# coco = train_data_loader.dataset.coco
#
# # get all image index info
# ids = list(sorted(coco.imgs.keys()))
# print("number of images: {}".format(len(ids)))
#
# # get all coco class labels
# coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
#
# import random
# random.seed(1)
# slice = random.sample(ids,1)
# # 遍历前三张图像
# for img_id in slice:
#     # 获取对应图像id的所有annotations idx信息
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#
#     # 根据annotations idx信息获取所有标注信息
#     targets = coco.loadAnns(ann_ids)
#
#     # get image file name
#     path = coco.loadImgs(img_id)[0]['file_name']
#
#     # read image
#     img = Image.open(os.path.join(img_path, path))
#     draw = ImageDraw.Draw(img)
#     # draw box to image
#     for target in targets:
#         x, y, w, h = target["bbox"]
#         x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
#         draw.rectangle((x1, y1, x2, y2),outline='red')
#         draw.text((x1, y1),coco_classes[target["category_id"]],fill='red',font=font1)
#
#     # show image
#     plt.imshow(img)
#     pylab.show()