_base_ = [
    'mosaic_quickdraw_grayscale.py'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMatAnnotations', target_key='CLASS_GT'),
    dict(type='CustomRerange', min_value=0., max_value=1.),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMatAnnotations', target_key='CLASS_GT'),
    dict(type='CustomRerange', min_value=0., max_value=1.),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline)
)

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline)
)

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline)
)
