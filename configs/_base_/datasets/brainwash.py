dataset_type = 'BrainwashDataset'
data_root = '/mnt/disk/brainwash/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu',
         transforms=[
             dict(
                 type='Crop',
                 x_min=0,
                 y_min=95,
                 x_max=640,
                 y_max=385,
                 p=1
             ),
             dict(
                 type='ShiftScaleRotate',
                 shift_limit=0.0625,
                 scale_limit=0.1,
                 rotate_limit=5,
                 border_mode=0,
                 interpolation=1,
                 p=1),
             dict(
                 type='RandomBrightnessContrast',
                 brightness_limit=[-0.1, 0.1],
                 contrast_limit=[-0.1, 0.1],
                 p=1)
         ],
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.3,
             min_area=100,
             filter_lost_elements=True),
         keymap={
             'img': 'image',
             'gt_masks': 'masks',
             'gt_bboxes': 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=True
         ),
    # dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'brainwash_train.idl',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'brainwash_val.idl',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'brainwash_val.idl',
        img_prefix=data_root,
        pipeline=test_pipeline)
)
evaluation = dict(interval=1, metric='mAP')
