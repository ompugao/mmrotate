# dataset settings
dataset_type = 'DOTADataset'
data_root = '/mnt/data/data/LOAF/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
classes=('person',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_512_dota/train/',
             img_prefix=data_root + 'images/resolution_512/train/',
             classes=classes,
             pipeline=train_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_1k_dota/train/',
             img_prefix=data_root + 'images/resolution_1k/train/',
             classes=classes,
             pipeline=train_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_2k_dota/train/',
             img_prefix=data_root + 'images/resolution_2k/train/',
             classes=classes,
             pipeline=train_pipeline),
        ],
        seperate_eval=False),
    val=dict(
        type='ConcatDataset',
        datasets=[
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_512_dota/val/',
             img_prefix=data_root + 'images/resolution_512/val/',
             classes=classes,
             pipeline=test_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_1k_dota/val/',
             img_prefix=data_root + 'images/resolution_1k/val/',
             classes=classes,
             pipeline=test_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_2k_dota/val/',
             img_prefix=data_root + 'images/resolution_2k/val/',
             classes=classes,
             pipeline=test_pipeline),
        ],
        seperate_eval=False),
    test=dict(
        type='ConcatDataset',
        datasets=[
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_512_dota/test/',
             img_prefix=data_root + 'images/resolution_512/test/',
             classes=classes,
             pipeline=test_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_1k_dota/test/',
             img_prefix=data_root + 'images/resolution_1k/test/',
             classes=classes,
             pipeline=test_pipeline),
        dict(type=dataset_type,
             ann_file=data_root + 'annotations/resolution_2k_dota/test/',
             img_prefix=data_root + 'images/resolution_2k/test/',
             classes=classes,
             pipeline=test_pipeline),
        ],
        seperate_eval=False)
    )

