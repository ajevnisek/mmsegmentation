import os
import shutil
import argparse

from tqdm import tqdm
from PIL import Image

import mmcv
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


CLASSES = ('original', 'augmented')
PALETTE = [[128, 128, 128], [255,20,147],]
GT_PALETTE = [[128, 128, 128], [144,238,144],]
INPUT_SIZE = 224


def convert_composite_image_name_to_mask_name(composite_image_name,
                                              seg_map_suffix='.png'):
    seg_map_filename_without_suffix = '_'.join(composite_image_name.split("_")[:2])
    seg_map = seg_map_filename_without_suffix + seg_map_suffix
    return seg_map


def convert_composite_image_name_to_real_image_name(composite_image_name,
                                              real_image_suffix='.jpg'):
    real_image_filename_without_suffix = composite_image_name.split("_")[0]
    real_image = real_image_filename_without_suffix + real_image_suffix
    return real_image



def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v_cut(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HCOCO", choices=['HCOCO',
                                                           'HAdobe5k',
                                                           'HFlickr',
                                                           'Hday2night'])
parser.add_argument('--is-cluster', default=False, action='store_true',
                    help="running on TAU cluster or on personal GPU.")
parser.add_argument('--process-all-images', default=False, action='store_true',
                    help='if set, then all images in the train and test sets '
                         'will be processed. Otherwise only 100 images.')
parser.add_argument('--longer-mask-prediction-training', default=False,
                    action='store_true',
                    help="Which cached results to use for prediction masks.")
args = parser.parse_args()

IMAGE_HARMONIZATION_DATASET = args.dataset

# IMAGE_HARMONIZATION_DATASET = 'HCOCO'

# is_cluster = False

# if is_cluster:
if args.is_cluster:
    data_root = os.path.join('/storage/jevnisek/ImageHarmonizationDataset/',
                             IMAGE_HARMONIZATION_DATASET)
    if args.longer_mask_prediction_training:
        target_root = '/storage/jevnisek/MaskPredictionDataset/longer_training'
    else:
        target_root = '/storage/jevnisek/MaskPredictionDataset/'
else:
    data_root = os.path.join(f'../data/Image_Harmonization_Dataset/',
                             IMAGE_HARMONIZATION_DATASET)
    target_root = os.path.join('../data/MaskPredictionDataset',)

os.makedirs(os.path.join(target_root,
                         IMAGE_HARMONIZATION_DATASET,), exist_ok=True)
os.makedirs(os.path.join(target_root,
                         IMAGE_HARMONIZATION_DATASET,
                         'real_images'),
            exist_ok=True)
os.makedirs(os.path.join(target_root,
                         IMAGE_HARMONIZATION_DATASET,
                         'composite_images'),
            exist_ok=True)

test_split = os.path.join(data_root, f'{IMAGE_HARMONIZATION_DATASET}_test.txt')
with open(test_split, 'r') as f:
    test_images = f.read().splitlines()

# if is_cluster:
if args.is_cluster:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_cluster.py')
else:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_personalGPU.py')
# if is_cluster:
if args.is_cluster:
    if args.longer_mask_prediction_training:
        checkpoint_file = os.path.join(
            '/storage/jevnisek/MaskPredictionCheckopoints/longer_training/',
            f'{IMAGE_HARMONIZATION_DATASET}.pth')
    else:
        checkpoint_file = os.path.join(
            '/storage/jevnisek/MaskPredictionCheckopoints/',
            f'{IMAGE_HARMONIZATION_DATASET}.pth')
else:
    checkpoint_file = os.path.join('checkpoints',
                                   f'{IMAGE_HARMONIZATION_DATASET}.pth')


print(f"LOADING checkpoint from: {checkpoint_file}")
print(f"SAVING masks to {target_root}")
d = torch.load(checkpoint_file)
d['meta']['PALETTE'] = PALETTE
torch.save(d, checkpoint_file)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

test_images.sort()

if args.process_all_images:
    train_split = os.path.join(data_root,
                              f'{IMAGE_HARMONIZATION_DATASET}_train.txt')
    with open(train_split, 'r') as f:
        train_images = f.read().splitlines()
    images_to_scan = train_images + test_images
else:
    images_to_scan = test_images[:100]

# test_image_name = test_images[0]
for test_image_name in tqdm(images_to_scan):
    img = os.path.join(data_root, 'composite_images', test_image_name)
    composite_image = Image.open(img)
    composite_image_tensor = ToTensor()(composite_image).unsqueeze(0)
    mask = os.path.join(data_root, 'masks',
                        convert_composite_image_name_to_mask_name(test_image_name))
    real = os.path.join(data_root, 'real_images',
                        convert_composite_image_name_to_real_image_name(
                            test_image_name))
    overlaid_ground_truth = model.show_result(
        img, (ToTensor()(Image.open(mask)) > 0.5).float(),
        palette=GT_PALETTE, show=False, opacity=0.5)
    overlaid_gt_pil = Image.fromarray(np.uint8(mmcv.bgr2rgb(overlaid_ground_truth))).convert('RGB')

    result = inference_segmentor(model, img)
    result = Resize((INPUT_SIZE, INPUT_SIZE))(torch.from_numpy(
        result[0].astype(np.uint8)).unsqueeze(0).unsqueeze(0))
    pil_result = Image.fromarray(result.squeeze(0).squeeze(0).numpy() * 255)
    pil_result.save(
        os.path.join(target_root, IMAGE_HARMONIZATION_DATASET,
                     'composite_images', os.path.basename(img)))

    result = inference_segmentor(model, real)
    result = Resize((INPUT_SIZE, INPUT_SIZE))(torch.from_numpy(
        result[0].astype(np.uint8)).unsqueeze(0).unsqueeze(0))
    pil_result = Image.fromarray(result.squeeze(0).squeeze(0).numpy() * 255)
    pil_result.save(
        os.path.join(target_root, IMAGE_HARMONIZATION_DATASET, 'real_images',
                     os.path.basename(real)))
