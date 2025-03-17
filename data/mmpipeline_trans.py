train_pipeline = [
    dict(type='LoadImageFromFilePyT'),
    dict(type='TopDownRandomFlip', flip_prob=0.2),
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=20, 
        scale_factor=0,
        rot_prob=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean = [102.9801, 115.9465, 122.7717],
        std =  [1.0, 1.0, 1.0],
        ),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='TopDownAffine',validation=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean = [102.9801, 115.9465, 122.7717],
        std =  [1.0, 1.0, 1.0],
    ),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = [
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean = [102.9801, 115.9465, 122.7717],
        std =  [1.0, 1.0, 1.0],
    ),
]