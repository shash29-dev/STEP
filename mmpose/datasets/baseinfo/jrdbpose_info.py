jrdbpose_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='right_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        2:
        dict(
            name='left_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='center_shoulder',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_elbow',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='center_hip',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        9:
        dict(
            name='right_wrist',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        10:
        dict(
            name='right_hip',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        
        12:
        dict(
            name='left_wrist',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        13:
        dict(
            name='right_knee',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        14:
        dict(
            name='left_knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        
        15:
        dict(
            name='right_foot',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='left_foot'),
        16:
        dict(
            name='left_foot',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='right_foot')
    },
    skeleton_info={
        0:
        dict(link=('head', 'center_shoulder'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('right_eye', 'left_eye'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_eye', 'right_eye'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_shoulder', 'left_shoulder'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('center_shoulder', 'center_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'right_shoulder'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_elbow', 'right_shoulder'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_elbow', 'left_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('center_hip', 'left_hip'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_wrist', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('right_hip', 'center_hip'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('left_hip', 'left_knee'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_wrist', 'left_elbow'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('right_knee', 'right_hip'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('left_knee', 'left_hip'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('right_foot', 'right_knee'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('left_foot', 'left_knee'), id=16, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])
