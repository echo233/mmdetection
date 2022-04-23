import transforms
import os
import json
import torch
from icme_dataset_2 import CocoDetection
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from tqdm import tqdm

def create_model(num_classes):

    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

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

    # create model
    model = create_model(num_classes=51)

    # load train weights
    weights_path = "/liu/code/d2l-zh/pycharm-local/faster_rcnn/multi_train/model_45.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    cpu_device = torch.device("cpu")

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    icme_result = []
    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(test_data_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            for target, output in zip(targets, outputs):
                output["boxes"][:, [2, 3]] = output["boxes"][:, [2, 3]] - output["boxes"][:, [0, 1]]
                boxes = output["boxes"].numpy().tolist()
                labels = output["labels"].numpy().tolist()
                scores = output["scores"].numpy().tolist()
                i = 0
                for score in scores:
                    if score > 0.1:
                        assert (len(boxes) == len(scores)), "{} error.".format(target["image_id"].item())
                        assert (len(boxes) == len(boxes)), "{} error.".format(target["image_id"].item())

                        icme_result.append({"image_id": target["image_id"].item(), "category_id": labels[i], "bbox": boxes[i], "score": scores[i]})
                        print("i is {}".format(i))
                    i =i +1

        with open("/liu/code/d2l-zh/pycharm-local/faster_rcnn/record_icme_2.json", 'w') as f:
            json.dump(icme_result, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='50', help='number of classes')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='/liu/icme/dataset/val/images/', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='/liu/code/d2l-zh/pycharm-local/faster_rcnn/multi_train/model_45.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when validation.')

    args = parser.parse_args()

    main(args)