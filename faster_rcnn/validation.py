"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""
from torchvision import models
models.resnet101(pretrained=)
import os
import json

import torch
import torchvision

from tqdm import tqdm
import numpy as np
from network_files import FasterRCNN, AnchorsGenerator
import transforms
from network_files import FasterRCNN
# from backbone import resnet50_fpn_backbone
#from my_dataset import VOCDataSet
from backbone import MobileNetV2
from icme_dataset_2 import CocoDetection
from train_utils import get_coco_api_from_dataset, CocoEvaluator


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = './coco91_indices.json'
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {v: k for k, v in class_dict.items()}


    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    test_dataset = CocoDetection("/liu/icme/dataset/", dataset="test", transforms=data_transform["val"], years="")

    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8,
                                             collate_fn=test_dataset.collate_fn)

    backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=51,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(test_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")
    icme_result = []
    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(test_data_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
            for target, output in zip(targets, outputs):
                boxes = output["boxes"].numpy().tolist()
                labels = output["labels"].numpy().tolist()
                scores = output["scores"].numpy().tolist()
                i = 0
                for score in scores:
                    if score > 0.3:
                        icme_result.append({"image_id": target["image_id"].item(), "category_id": labels[i], "bbox": boxes[i], "score": scores[i]})
                        print("i is {}".format(i))
                    i =i +1

        with open("/liu/code/d2l-zh/pycharm-local/faster_rcnn/record_icme_2.json", 'w') as f:
            json.dump(icme_result, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='50', help='number of classes')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='/liu/icme/dataset/val/images/', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='/liu/code/d2l-zh/pycharm-local/faster_rcnn/save_weights/mobile-model-24.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)
