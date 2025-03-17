mousepose_info = dict(
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
        dict(name='snout', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='leftear',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='rightear',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        4:
        dict(
            name='spine1',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        5:
        dict(
            name='spine2',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        6:
        dict(
            name='spine3',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        7:
        dict(
            name='spine4',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        11:
        dict(
            name='tailbase',
            id=4,
            color=[0,255,255],
            type='upper',
            swap=''),
        8:
        dict(
            name='tail1',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        9:
        dict(
            name='tail2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        10:
        dict(
            name='tail3',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('leftear', 'snout'), id=1, color=[0, 255, 0]),
        1:
        dict(link=('rightear', 'snout'), id=2, color=[0, 255, 0]),
        2:
        dict(link=('snout', 'shoulder'), id=0, color=[255, 128, 0]),
        3:
        dict(link=('shoulder', 'spine1'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('spine1', 'spine2'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('spine2', 'spine3'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('spine3', 'spine4'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('spine4', 'tail1'), id=7, color=[53,123,255]),
        8:
        dict(link=('tail1', 'tail2'), id=8, color=[53,123,255]),
        9:
        dict(link=('tail2', 'tail3'), id=9, color=[53,123,255]),
        10:
        dict(link=('tail3', 'tailbase'), id=10, color=[53,123,255]),
        
        
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.026, 0.025, 0.025, 0.035, 0.035, 0.026, 0.025,
    ])
