import os
import shutil
import argparse

from tqdm import tqdm
from PIL import Image

import mmcv
import torch
import numpy as np
from torchvision.transforms import ToTensor

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


CLASSES = ('original', 'augmented')
PALETTE = [[128, 128, 128], [255,20,147],]
GT_PALETTE = [[128, 128, 128], [144,238,144],]


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
parser.add_argument('--all-test-images', default=False, action='store_true',
                    help='if set, then all images in the test set will be '
                         'tested. Otherwise only 100 images.')

args = parser.parse_args()

IMAGE_HARMONIZATION_DATASET = args.dataset

# IMAGE_HARMONIZATION_DATASET = 'HCOCO'

# is_cluster = False

# if is_cluster:
if args.is_cluster:
    data_root = os.path.join('/storage/jevnisek/ImageHarmonizationDataset/',
                             IMAGE_HARMONIZATION_DATASET)
    harmonized_by_harmonizer_data_root = os.path.join(
        '/storage/jevnisek/HarmonizerData/',
        IMAGE_HARMONIZATION_DATASET)
    target_root = '/storage/jevnisek/MaskPredictionOnHarmonizer' \
                  '/mask_prediction_on_harmonizer_output_results/'
else:
    data_root = os.path.join(f'../data/Image_Harmonization_Dataset/',
                             IMAGE_HARMONIZATION_DATASET)
    harmonized_by_harmonizer_data_root = os.path.join(
        f'../Harmonizer/dataset/harmonized_images/',
        IMAGE_HARMONIZATION_DATASET)
    target_root = os.path.join('mask_prediction_on_harmonizer_output_results',)

os.makedirs(os.path.join(target_root,
                         IMAGE_HARMONIZATION_DATASET,), exist_ok=True)

test_images = os.listdir(harmonized_by_harmonizer_data_root)
test_images.sort()
# if is_cluster:
if args.is_cluster:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_cluster.py')
else:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_personalGPU.py')

if args.is_cluster:
    checkpoint_file = os.path.join(
        '/storage/jevnisek/MaskPredictionCheckopoints/',
        f'{IMAGE_HARMONIZATION_DATASET}.pth')
else:
    checkpoint_file = os.path.join('checkpoints',
                                   f'{IMAGE_HARMONIZATION_DATASET}.pth')

d = torch.load(checkpoint_file)
d['meta']['PALETTE'] = PALETTE
torch.save(d, checkpoint_file)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

images_to_scan = test_images if args.all_test_images else test_images[:100]
# test_image_name = test_images[0]
for test_image_name in tqdm(images_to_scan):
    img = os.path.join(harmonized_by_harmonizer_data_root, test_image_name)
    harmonized_by_harmonizer = Image.open(img)
    harmonized_by_harmonizer_tensor = ToTensor()(harmonized_by_harmonizer).unsqueeze(0)
    mask = os.path.join(data_root, 'masks',
                        convert_composite_image_name_to_mask_name(test_image_name))
    real = os.path.join(data_root, 'real_images',
                        convert_composite_image_name_to_real_image_name(
                            test_image_name))
    overlaid_ground_truth = model.show_result(img, ToTensor()(Image.open(mask)),
                                              palette=GT_PALETTE, show=False,
                                              opacity=0.5)
    overlaid_gt_pil = Image.fromarray(np.uint8(mmcv.bgr2rgb(overlaid_ground_truth))).convert('RGB')

    result = inference_segmentor(model, img)
    overlaid_img = model.show_result(img, result,
                                     palette=PALETTE, show=False, opacity=0.5)
    overlaid_image_rgb = mmcv.bgr2rgb(overlaid_img)
    overlaid_pil_image_rgb = Image.fromarray(np.uint8(overlaid_image_rgb)).convert('RGB')
    """
    Final image will be:
    +++++++++++++++++++++++++++++++++++++++++++++++
    + real          | generated by Harmonizer     +
    + ground truth  | estimated map               +
    +++++++++++++++++++++++++++++++++++++++++++++++
    """
    concatenated_image_top = get_concat_h(Image.open(real),
                                          Image.open(img))
    concatenated_image_bottom = get_concat_h(overlaid_gt_pil,
                                             overlaid_pil_image_rgb)
    concatenated_image = get_concat_v_cut(concatenated_image_top,
                                          concatenated_image_bottom)
    concatenated_image.resize((1024, 1024)).save(os.path.join(
        target_root,
        IMAGE_HARMONIZATION_DATASET,
        f"{test_image_name[:-4]}_on_harmonizer.jpg"))

