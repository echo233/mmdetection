_base_ = 'split1_shot1.py'

split = 1
shot = 10
data = dict(train=dict(split=split, shot=shot),
            val=dict(split=split),
            test=dict(split=split))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1*4,
                 warmup_ratio=0.001,
                 step=[16000*4])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=20000*4)

checkpoint_config = dict(interval=2000*4)
evaluation = dict(interval=2000*4)
