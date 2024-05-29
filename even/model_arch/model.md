start training
{'dataset': {'crop_ratio': [0.9, 1.0],
             'default_fps': 30,
             'downsample_rate': 1,
             'feat_folder': './data/epic_kitchens/features',
             'feat_stride': 16,
             'file_ext': '.npz',
             'file_prefix': None,
             'force_upsampling': False,
             'input_dim': 2304,
             'json_file': './data/epic_kitchens/annotations/epic_kitchens_100_verb.json',
             'max_seq_len': 2304,
             'num_classes': 97,
             'num_frames': 32,
             'trunc_thresh': 0.3},
 'dataset_name': 'epic',
 'devices': ['cuda:0'],
 'init_rand_seed': 1234567891,
 'loader': {'batch_size': 2, 'num_workers': 4},
 'model': {'backbone_arch': (2, 2, 5),
           'backbone_type': 'SGP',
           'boudary_kernel_size': 3,
           'downsample_type': 'max',
           'embd_dim': 512,
           'embd_kernel_size': 3,
           'embd_with_ln': True,
           'fpn_dim': 512,
           'fpn_type': 'identity',
           'fpn_with_ln': True,
           'head_dim': 512,
           'head_kernel_size': 3,
           'head_num_layers': 3,
           'head_with_ln': True,
           'init_conv_vars': 0,
           'input_dim': 2304,
           'input_noise': 0,
           'iou_weight_power': 0.25,
           'k': 2,
           'max_buffer_len_factor': 4.0,
           'max_seq_len': 2304,
           'n_sgp_win_size': 5,
           'num_bins': 16,
           'num_classes': 97,
           'regression_range': [[0, 4],
                                [2, 8],
                                [4, 16],
                                [8, 32],
                                [16, 64],
                                [32, 10000]],
           'scale_factor': 2,
           'sgp_mlp_dim': 1024,
           'test_cfg': {'duration_thresh': 0.05,
                        'ext_score_file': None,
                        'iou_threshold': 0.1,
                        'max_seg_num': 2000,
                        'min_score': 0.001,
                        'multiclass_nms': True,
                        'nms_method': 'soft',
                        'nms_sigma': 0.4,
                        'pre_nms_thresh': 0.001,
                        'pre_nms_topk': 5000,
                        'voting_thresh': 0.75},
           'train_cfg': {'center_sample': 'radius',
                         'center_sample_radius': 1.5,
                         'clip_grad_l2norm': 1.0,
                         'cls_prior_prob': 0.01,
                         'dropout': 0.0,
                         'droppath': 0.1,
                         'head_empty_cls': [],
                         'init_loss_norm': 250,
                         'label_smoothing': 0.0,
                         'loss_weight': 1.0},
           'use_abs_pe': False,
           'use_trident_head': True},
 'model_name': 'TriDet',
 'opt': {'epochs': 22,
         'eta_min': 5e-05,
         'learning_rate': 0.0001,
         'momentum': 0.9,
         'schedule_gamma': 0.1,
         'schedule_steps': [],
         'schedule_type': 'cosine',
         'type': 'AdamW',
         'warmup': True,
         'warmup_epochs': 5,
         'weight_decay': 0.05},
 'output_folder': './ckpt/',
 'test_cfg': {'duration_thresh': 0.05,
              'ext_score_file': None,
              'iou_threshold': 0.1,
              'max_seg_num': 2000,
              'min_score': 0.001,
              'multiclass_nms': True,
              'nms_method': 'soft',
              'nms_sigma': 0.4,
              'pre_nms_thresh': 0.001,
              'pre_nms_topk': 5000,
              'voting_thresh': 0.75},
 'train_cfg': {'center_sample': 'radius',
               'center_sample_radius': 1.5,
               'clip_grad_l2norm': 1.0,
               'cls_prior_prob': 0.01,
               'dropout': 0.0,
               'droppath': 0.1,
               'head_empty_cls': [],
               'init_loss_norm': 250,
               'label_smoothing': 0.0,
               'loss_weight': 1.0},
 'train_split': ['training'],
 'val_split': ['validation']}
TriDet(
  (backbone): SGPBackbone(
    (relu): ReLU(inplace=True)
    (embd): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(2304, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (embd_norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (stem): ModuleList(
      (0): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): Identity()
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): Identity()
        (drop_path_mlp): Identity()
        (act): GELU(approximate=none)
      )
      (1): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): Identity()
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): Identity()
        (drop_path_mlp): Identity()
        (act): GELU(approximate=none)
      )
    )
    (branch): ModuleList(
      (0): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(13,), stride=(1,), padding=(6,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (1): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(13,), stride=(1,), padding=(6,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (2): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(13,), stride=(1,), padding=(6,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (3): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(13,), stride=(1,), padding=(6,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (4): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(5,), stride=(1,), padding=(2,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(13,), stride=(1,), padding=(6,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(1024, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
    )
  )
  (neck): FPNIdentity(
    (fpn_norms): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
      (2): LayerNorm()
      (3): LayerNorm()
      (4): LayerNorm()
      (5): LayerNorm()
    )
  )
  (point_generator): PointGenerator(
    (buffer_points): BufferList()
  )
  (cls_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 97, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (start_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 97, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (end_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 97, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (reg_head): RegHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (scale): ModuleList(
      (0): Scale()
      (1): Scale()
      (2): Scale()
      (3): Scale()
      (4): Scale()
      (5): Scale()
    )
    (offset_head): MaskedConv1D(
      (conv): Conv1d(512, 34, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
)


==============================================================================
start training
{'dataset': {'crop_ratio': [0.9, 1.0],
             'default_fps': None,
             'downsample_rate': 1,
             'feat_folder': './data/thumos/i3d_features',
             'feat_stride': 4,
             'file_ext': '.npy',
             'file_prefix': None,
             'force_upsampling': False,
             'input_dim': 2048,
             'json_file': './data/thumos/annotations/thumos14.json',
             'max_seq_len': 2304,
             'num_classes': 20,
             'num_frames': 16,
             'trunc_thresh': 0.5},
 'dataset_name': 'thumos',
 'devices': ['cuda:0'],
 'init_rand_seed': 1234567891,
 'loader': {'batch_size': 2, 'num_workers': 4},
 'model': {'backbone_arch': [2, 2, 5],
           'backbone_type': 'SGP',
           'boudary_kernel_size': 3,
           'downsample_type': 'max',
           'embd_dim': 512,
           'embd_kernel_size': 3,
           'embd_with_ln': True,
           'fpn_dim': 512,
           'fpn_type': 'identity',
           'fpn_with_ln': True,
           'head_dim': 512,
           'head_kernel_size': 3,
           'head_num_layers': 3,
           'head_with_ln': True,
           'init_conv_vars': 0,
           'input_dim': 2048,
           'input_noise': 0.0005,
           'iou_weight_power': 0.2,
           'k': 5,
           'max_buffer_len_factor': 6.0,
           'max_seq_len': 2304,
           'n_sgp_win_size': 1,
           'num_bins': 16,
           'num_classes': 20,
           'regression_range': [[0, 4],
                                [4, 8],
                                [8, 16],
                                [16, 32],
                                [32, 64],
                                [64, 10000]],
           'scale_factor': 2,
           'sgp_mlp_dim': 768,
           'test_cfg': {'duration_thresh': 0.05,
                        'ext_score_file': None,
                        'iou_threshold': 0.1,
                        'max_seg_num': 2000,
                        'min_score': 0.001,
                        'multiclass_nms': True,
                        'nms_method': 'soft',
                        'nms_sigma': 0.5,
                        'pre_nms_thresh': 0.001,
                        'pre_nms_topk': 2000,
                        'voting_thresh': 0.7},
           'train_cfg': {'center_sample': 'radius',
                         'center_sample_radius': 1.5,
                         'clip_grad_l2norm': 1.0,
                         'cls_prior_prob': 0.01,
                         'dropout': 0.0,
                         'droppath': 0.1,
                         'head_empty_cls': [],
                         'init_loss_norm': 100,
                         'label_smoothing': 0.0,
                         'loss_weight': 1.0},
           'use_abs_pe': False,
           'use_trident_head': True},
 'model_name': 'TriDet',
 'opt': {'epochs': 20,
         'eta_min': 1e-08,
         'learning_rate': 0.0001,
         'momentum': 0.9,
         'schedule_gamma': 0.1,
         'schedule_steps': [],
         'schedule_type': 'cosine',
         'type': 'AdamW',
         'warmup': True,
         'warmup_epochs': 20,
         'weight_decay': 0.025},
 'output_folder': './ckpt/',
 'test_cfg': {'duration_thresh': 0.05,
              'ext_score_file': None,
              'iou_threshold': 0.1,
              'max_seg_num': 2000,
              'min_score': 0.001,
              'multiclass_nms': True,
              'nms_method': 'soft',
              'nms_sigma': 0.5,
              'pre_nms_thresh': 0.001,
              'pre_nms_topk': 2000,
              'voting_thresh': 0.7},
 'train_cfg': {'center_sample': 'radius',
               'center_sample_radius': 1.5,
               'clip_grad_l2norm': 1.0,
               'cls_prior_prob': 0.01,
               'dropout': 0.0,
               'droppath': 0.1,
               'head_empty_cls': [],
               'init_loss_norm': 100,
               'label_smoothing': 0.0,
               'loss_weight': 1.0},
 'train_split': ['validation'],
 'val_split': ['test']}
TriDet(
  (backbone): SGPBackbone(
    (relu): ReLU(inplace=True)
    (embd): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(2048, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (embd_norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (stem): ModuleList(
      (0): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): Identity()
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): Identity()
        (drop_path_mlp): Identity()
        (act): GELU(approximate=none)
      )
      (1): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): Identity()
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): Identity()
        (drop_path_mlp): Identity()
        (act): GELU(approximate=none)
      )
    )
    (branch): ModuleList(
      (0): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (1): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (2): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (3): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
      (4): SGPBlock(
        (ln): LayerNorm()
        (gn): GroupNorm(16, 512, eps=1e-05, affine=True)
        (psi): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convw): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (convkw): Conv1d(512, 512, kernel_size=(11,), stride=(1,), padding=(5,), groups=512)
        (global_fc): Conv1d(512, 512, kernel_size=(1,), stride=(1,), groups=512)
        (downsample): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (mlp): Sequential(
          (0): Conv1d(512, 768, kernel_size=(1,), stride=(1,))
          (1): GELU(approximate=none)
          (2): Conv1d(768, 512, kernel_size=(1,), stride=(1,))
        )
        (drop_path_out): AffineDropPath()
        (drop_path_mlp): AffineDropPath()
        (act): GELU(approximate=none)
      )
    )
  )
  (neck): FPNIdentity(
    (fpn_norms): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
      (2): LayerNorm()
      (3): LayerNorm()
      (4): LayerNorm()
      (5): LayerNorm()
    )
  )
  (point_generator): PointGenerator(
    (buffer_points): BufferList()
  )
  (cls_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 20, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (start_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 20, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (end_head): ClsHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (cls_head): MaskedConv1D(
      (conv): Conv1d(512, 20, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (reg_head): RegHead(
    (act): ReLU()
    (head): ModuleList(
      (0): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
      (1): MaskedConv1D(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      )
    )
    (norm): ModuleList(
      (0): LayerNorm()
      (1): LayerNorm()
    )
    (scale): ModuleList(
      (0): Scale()
      (1): Scale()
      (2): Scale()
      (3): Scale()
      (4): Scale()
      (5): Scale()
    )
    (offset_head): MaskedConv1D(
      (conv): Conv1d(512, 34, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
)