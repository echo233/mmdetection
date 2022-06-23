_base_ = '../_base_/faster_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['fsodss.models.cosine_sim_bbox_head', 'fsodss.datasets.few_shot_voc', 'fsodss.models.tri_single_level_roi_extractor'])


model = dict(pretrained=None,
             backbone=dict(depth=101, frozen_stages=4),
             rpn_head=dict(anchor_generator=dict(scale_major=False)),
             roi_head=dict(bbox_roi_extractor=dict(type='TriSingleRoIExtractor',),# SingleRoIExtractor修改了
                           bbox_head=dict(type='CosineSimBBoxHead',  # 新增的，继承TFA的cos头
                                          num_classes=20,
                                          loss_bbox=dict(loss_weight=0.))))

unfreeze_layers = ('fc_cls', 'fc_reg')
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
#                     std=[58.395, 57.12, 57.375],
#                     to_rgb=False)

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], # caffe是这个参数
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                    (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                    (1333, 736), (1333, 768), (1333, 800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

split = 1
shot = 1

dataset_type = 'VOCDataset'
data_root = '/liu/code/d2l-zh/pycharm-local/data/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotVOCDataset',  # 新增的类
        ann_file=None,  # not used in few shot voc
        split=split,
        shot=shot,
        img_prefix=None,
        pipeline=train_pipeline),
    val=dict(type='FewShotVOCTestDataset',  # 新增的类
             ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
             split=split,
             img_prefix=data_root + 'VOC2007/',
             pipeline=test_pipeline),
    test=dict(type='FewShotVOCTestDataset',
              ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
              split=split,
              img_prefix=data_root + 'VOC2007/',
              pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.001/4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1*4,
                 warmup_ratio=0.001,
                 step=[3000*4])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=4000*4)

checkpoint_config = dict(interval=1000*4)
evaluation = dict(interval=4000*4, metric='mAP')

load_from = '/liu/code/d2l-zh/pycharm-local/FADI-Main/models/voc_split1_base.pth'
#load_from = '/liu/code/d2l-zh/pycharm-local/FSODSS/tools/work_dirs/split1_shot1/latest.pth'
