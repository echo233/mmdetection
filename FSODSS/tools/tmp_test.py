import numpy as np
import random
import cv2
import os
import xml.etree.ElementTree as ET

ann_path = '/liu/code/d2l-zh/pycharm-local/data/VOCdevkit/VOC2007/Annotations/'
# 想法：通过语义相关性，把最类似的图像沾到原始位置，
class copy_paste(object):
    def __init__(self, thresh=32*32, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        self.thresh = thresh
        self.prob = prob,
        self.copy_time = copy_times
        self.epochs = epochs
        self.all_object = all_objects
        self.one_object = one_object

    def issmallobject(self, h, w):
        print(h)
        print(w)
        print(self.thresh)
        if h*w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
        if annot_a is None: return False
        left_max = max(annot_a[1], annot_b[1])
        top_max = max(annot_a[2], annot_b[2])
        right_min = min(annot_a[3], annot_b[3])
        bottom_min = min(annot_a[4], annot_b[4])
        inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, new_l, labels):
        for l in labels:
            if self.compute_overlap(new_l, l): return False
        return True

    def create_copy_label(self, h, w, l, labels):  # 参数为图像的宽高，粘贴对象的标注，整个图像的标注
        l = l.astype(np.int)
        l_h, l_w = l[4] - l[2], l[3] - l[1]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(l_w / 2), int(w - l_w / 2)), \
                                 np.random.randint(int(l_h / 2), int(h - l_h / 2))  # 随机找一个位置放
            xmin, ymin = random_x - l_w / 2, random_y - l_h / 2
            xmax, ymax = xmin + l_w, ymin + l_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_l = np.array([l[0], xmin, ymin, xmax, ymax]).astype(np.int)
            if self.donot_overlap(new_l, labels) is False:
                continue
            return new_l
        return None


    def add_patch_in_img(self, new_label, l, image):
        l = l.astype(np.int)
        image[new_label[2]:new_label[4], new_label[1]:new_label[3], :] = image[l[2]:l[4], l[1]:l[3], :]
        return image

    def __call__(self, image, labels):
        """
        image: numpy.ndarry (1280, 1280, 3)
        labels: [:, class+xyxy] 没用归一化的  numpy.ndarry (6, 5)
        """
        h, w = image.shape[0]/2, image.shape[1]/2
        small_object_list = []
        for i in range(labels.shape[0]):
            label = labels[i]
            label_h, label_w = label[4] - label[2], label[3] - label[1]
            if self.issmallobject(label_h, label_w):
                small_object_list.append(i)
        l = len(small_object_list)
        if l == 0: return image, labels

        # 随机copy的个数
        copy_object_num = np.random.randint(0, l)
        # 复制所有
        if self.all_object:
            copy_object_num = l
        if self.one_object:
            copy_object_num = 1

        # 在 0~l-1 之间随机取copy_object_num个数
        random_list = random.sample(range(l), copy_object_num)  # list[2,1,8]
        label_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_label = labels[label_idx_of_small_object, :]

        for idx in range(copy_object_num):
            l = select_label[idx]
            l_h, l_w = l[4] - l[2], l[3] - l[1]
            if self.issmallobject(l_h, l_w) is False: continue

            for i in range(self.copy_time):
                new_label = self.create_copy_label(h, w, l, labels)  # 粘贴的新对象标注
                if new_label is not None:
                    image = self.add_patch_in_img(new_label, l, image)  # 更新原始图像中粘贴的对象处的信息
                    labels = np.append(labels, new_label.reshape(1, -1), axis=0)  # 更新原始图像的标注信息

        return image, labels

def get_simClass(fake_name):
    items = []
    for f in os.listdir(ann_path):
        tree = ET.parse(ann_path+f)
        root = tree.getroot()
        result = root.findall('object')
        bool_num = 0
        for obj in result:
            if obj.find('name').text == fake_name:
                items.append(f.split('.')[0])
            # else:
            #     root.remove(obj)  # 需要移除吗
    return items

def create_copy_label()
def main():
    fake_name ='dog'
    class_item = '/liu/code/d2l-zh/pycharm-local/data/VOCdevkit/VOC2007/ImageSets/Main/' + fake_name + '_trainval.txt'  # 从其中随机取一些
    items = get_simClass(fake_name)

    # < size >
    #     < width > 500 < / width >
    #     < height > 333 < / height >
    #     < depth > 3 < / depth >
    # < / size >
    random_list = random.sample(range(len(items)), min(len(items),10)) # 取100个该类别的目标,返回值是索引
    for i in random_list:
        ann_path+items[i]+'.xml'
        tree = ET.parse(ann_path+items[i]+'.xml')

        for i, obj in enumerate(tree.findall('object')):
            r = {
                'image_id': items[i],
                'height': int(tree.findall('./size/height')[0].text),
                'width': int(tree.findall('./size/width')[0].text),
                'idx': i,
            }
            cls_ = obj.find('name').text
            bbox = obj.find('bndbox')
            bbox = [
                float(bbox.find(x).text)
                for x in ['xmin', 'ymin', 'xmax', 'ymax']
            ]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            # instances = [{
            #     'category_id': classnames.index(cls),
            #     'bbox': bbox,
            # }]
            # r['annotations'] = instances
            # dicts_.append(r)


    h, w, l, labels = 800, 800,  # 原始图片信息都可以得到,l原图左上右下和标签，labels新图像中对象的位置
    create_copy_label(h, w, l, labels)
    print(1)
    # image = np.ndarray((128,128,3))
    # labels = np.ones((6,5))
    # ttt = copy_paste(thresh=32*32, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False)
    # ttt(image,labels)
    # random_list = random.sample(range(10), 3)
    # print(random_list)
if __name__ == '__main__':
    main()

