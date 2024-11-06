_base_ = [
    '../_base_/runtime/default_runtime.py',
    '../_base_/datasets/sky_scene_rgb.py',
    '../../configs/_base_/models/segformer_mit-b0.py',
    '../_base_/schedules/schedule_by_epoch_finetune.py'
]

data_preprocessor = dict(_delete_=True, type='CustomDataPreProcessor')

checkpoint = 'Checkpoints/sketchseger_init.pth'

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformerWithLocalDetail', embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 18, 3], with_local_detail=True),
    decode_head=dict(type='MDFHeadWithLocalDetail', image_size=512, in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3], num_classes=30, with_local_detail=True, mdf_block_num=2,
                     loss_decode=dict(type='CustomCrossEntropyLoss', label_smoothing=0.1))
)

work_dir='work_dirs/sketchseger_sky_scene'

train_dataloader = dict(batch_size=1, num_workers=16)
val_dataloader = dict(batch_size=1, num_workers=16)
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    dataset=dict(
        data_prefix=dict(img_path='test/DRAWING_GT', seg_map_path='test/CLASS_GT')
    )
)

train_cfg = dict(val_interval=2)
