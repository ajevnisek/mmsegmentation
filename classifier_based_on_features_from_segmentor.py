import torch
from mmcv import Config
from mmseg.models import build_segmentor

config_file_path = 'configs/HFlickr_personalGPU.py'
checkpoint_path = 'checkpoints/longer_training/HFlickr.pth'

cfg = Config.fromfile(config_file_path)
model = build_segmentor(cfg.model)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
