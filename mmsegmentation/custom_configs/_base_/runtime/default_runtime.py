# custom import
custom_imports = dict(imports=['custommmseg'], allow_failed_imports=False)
# runtime setting
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
log_level = 'INFO'
launcher = 'pytorch'
randomness=dict(seed=0,
                deterministic=False,
                diff_rank_seed=False)
load_from = None
resume = False
