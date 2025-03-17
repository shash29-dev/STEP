marmosetpose_info = dict(
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
        dict(name='front', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='right',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='middle',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='left',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        4:
        dict(
            name='fl1',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        5:
        dict(
            name='bl1',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        6:
        dict(
            name='fr1',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        7:
        dict(
            name='br1',
            id=4,
            color=[0,255,0],
            type='upper',
            swap=''),
        8:
        dict(
            name='bl2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        9:
        dict(
            name='br2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        10:
        dict(
            name='fl2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        11:
        dict(
            name='fr2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        12:
        dict(
            name='body1',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        13:
        dict(
            name='body2',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
        14:
        dict(
            name='body3',
            id=4,
            color=[255,128,0],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('right', 'front'), id=1, color=[0, 255, 0]),
        1:
        dict(link=('left', 'front'), id=3, color=[0, 255, 0]),
        2:
        dict(link=('front', 'middle'), id=0, color=[255, 128, 0]),
        3:
        dict(link=('left', 'fl1'), id=2, color=[255, 128, 0]),
        4:
        dict(link=('fr1', 'body2'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('body2', 'body3'), id=10, color=[255, 128, 0]),
        6:
        dict(link=('fl1', 'bl1'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('bl1', 'fr1'), id=11, color=[53,123,255]),
        8:
        dict(link=('fl1', 'br2'), id=12, color=[53,123,255]),
        9:
        dict(link=('fr1', 'fr2'), id=13, color=[53,123,255]),
        10:
        dict(link=('br2', 'fl2'), id=4, color=[255, 128, 0]),
        11:
        dict(link=('fl1', 'br1'), id=5, color=[255, 128, 0]),
        12:
        dict(link=('br1', 'bl2'), id=6, color=[255, 128, 0]),
        13:
        dict(link=('fr2', 'body1'), id=7, color=[53,123,255]),
        
        
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. , 1.,
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.026, 0.025, 0.025, 0.035, 0.035, 0.026, 0.025, 0.026, 0.025, 0.025,
    ])
