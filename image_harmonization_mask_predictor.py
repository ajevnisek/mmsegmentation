import mmseg
import torch, torchvision

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
# Let's take a look at the dataset
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv.utils import print_log
from mmseg.utils import get_root_logger

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HCOCO", choices=['HCOCO',
                                                           'HAdobe5k',
                                                           'HFlickr',
                                                           'Hday2night'])
DATASETS_TRAIN_SIZE = {'HCOCO': 38545,
                       'HAdobe5k': 19437,
                       'HFlickr': 7449,
                       'Hday2night': 311}
args = parser.parse_args()
IMAGE_HARMONIZATION_DATASET = args.dataset



@DATASETS.register_module()
class ImageHarmonizationMasksDataset(CustomDataset):
    CLASSES = ('original', 'augmented')
    PALETTE = ([129, 127, 38], [0, 11, 123])

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        lines = mmcv.list_from_file(
            split, file_client_args=self.file_client_args)
        for line in lines:
            img_name = line.strip()
            img_info = dict(filename=img_name)
            if ann_dir is not None:
                seg_map_filename_without_suffix = '_'.join(img_name.split("_")[:2])
                seg_map = seg_map_filename_without_suffix + seg_map_suffix
                img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


data_root = f'../data/Image_Harmonization_Dataset/{IMAGE_HARMONIZATION_DATASET}'
img_dir = 'composite_images'
ann_dir = 'masks'
from mmcv import Config
cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py')
from mmseg.apis import set_random_seed
from mmseg.utils import get_device

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2
cfg.model.auxiliary_head.num_classes = 2

# Modify dataset type and path
cfg.dataset_type = 'ImageHarmonizationMasksDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu=8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (512, 512)
cfg.train_pipeline = [{'type': 'LoadImageFromFile'},
                      dict(type='LoadAnnotations',
                           is_image_harmonization_dataset=True,
                           image_coversion_flag='grayscale'),
 {'type': 'Resize', 'img_scale': (512, 512)},
 {'type': 'RandomFlip', 'prob': 0.5},
 {'type': 'PhotoMetricDistortion'},
 {'type': 'Normalize',
  'mean': [123.675, 116.28, 103.53],
  'std': [58.395, 57.12, 57.375],
  'to_rgb': True},
 {'type': 'Pad', 'size': (512, 512), 'pad_val': 0, 'seg_pad_val': 255},
 {'type': 'DefaultFormatBundle'},
 {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]

cfg.test_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'MultiScaleFlipAug',
  'img_scale': (512, 512),
  'flip': False,
  'transforms': [{'type': 'Resize', 'keep_ratio': True},
   {'type': 'RandomFlip'},
   {'type': 'Normalize',
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375],
    'to_rgb': True},
   {'type': 'ImageToTensor', 'keys': ['img']},
   {'type': 'Collect', 'keys': ['img']}]}]



cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = f'{IMAGE_HARMONIZATION_DATASET}_train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = f'{IMAGE_HARMONIZATION_DATASET}_test.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = f'{IMAGE_HARMONIZATION_DATASET}_test.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'checkpoints/pspnet_r50-d8_512x512_160k_ade20k_20200615_184358-1890b0bd.pth'

# Set up working dir to save files and logs.
cfg.work_dir = f'./work_dirs/{IMAGE_HARMONIZATION_DATASET}'

# cfg.runner.max_iters = min(
#     10000, 15 * DATASETS_TRAIN_SIZE[IMAGE_HARMONIZATION_DATASET])
cfg.runner.max_iters = 40
cfg.log_config.interval = 10
cfg.evaluation.interval = 20
cfg.checkpoint_config.interval = 200
cfg.evaluation.is_image_harmonization_dataset = True

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor


# Build the dataset
datasets = [build_dataset(cfg.data.train)]
sample = datasets[0][0]



# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())
from mmseg.apis.test import single_gpu_test
from torch.utils.data import DataLoader
single_gpu_test(model=model,
                data_loader=DataLoader(build_dataset(cfg.data.test), batch_size=8),
                show=True,
                out_dir=osp.join(cfg.work_dir, 'test'),
                efficient_test=False,
                opacity=0.5,
                pre_eval=False,
                format_only=False,
                format_args={})