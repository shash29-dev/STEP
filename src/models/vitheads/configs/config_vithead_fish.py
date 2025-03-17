

channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[
        [0, 1, 2, 3, 4,],
    ],
    inference_channel=[
        0, 1, 2, 3, 4,
    ])

config_vithead_fish = dict(
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=256,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        upsample=0,
        in_index=0,
        align_corners=False,
        input_transform=None,
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11),
)