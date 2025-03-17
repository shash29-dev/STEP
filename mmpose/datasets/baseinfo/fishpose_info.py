fishpose_info = dict(
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
        dict(name='tip', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='gill',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='peduncle',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='dorsal',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        4:
        dict(
            name='caudal',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('tip', 'gill'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('gill', 'peduncle'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('peduncle', 'dorsal'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('dorsal', 'caudal'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('caudal', 'dorsal'), id=4, color=[51, 153, 255]),
        
    },
    joint_weights=[
        1., 1., 1., 1., 1., 
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035,
    ])
