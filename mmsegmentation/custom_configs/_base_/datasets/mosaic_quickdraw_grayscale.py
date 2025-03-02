# dataset settings
dataset_type = 'MosaicQuickdrawDataset'
data_root = 'Datasets/MosaicQuickdraw'

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadMatAnnotations', target_key='CLASS_GT'),
    dict(type='CustomRerange', min_value=0., max_value=1.),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadMatAnnotations', target_key='CLASS_GT'),
    dict(type='CustomRerange', min_value=0., max_value=1.),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='sketch/train', seg_map_path='annotation/train'),
        pipeline=train_pipeline),
    collate_fn=dict(type='default_collate'),
    drop_last=True
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='sketch/val', seg_map_path='annotation/val'),
        pipeline=test_pipeline),
    collate_fn=dict(type='default_collate'),
    drop_last=True
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='sketch/test', seg_map_path='annotation/test'),
        pipeline=test_pipeline),
    collate_fn=dict(type='default_collate'),
    drop_last=True
)

val_evaluator = dict(type='SketchIoUMetric', iou_metrics=['fwIoU'], ignore_index=255, nan_to_num=0)
test_evaluator = val_evaluator
