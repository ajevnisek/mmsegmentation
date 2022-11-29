import os
import shutil
import argparse

from tqdm import tqdm
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


CLASSES = ('original', 'augmented')
PALETTE = [[128, 128, 128], [255,20,147],]


def convert_composite_image_name_to_mask_name(composite_image_name,
                                              seg_map_suffix='.png'):
    seg_map_filename_without_suffix = '_'.join(composite_image_name.split("_")[:2])
    seg_map = seg_map_filename_without_suffix + seg_map_suffix
    return seg_map


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="HCOCO", choices=['HCOCO',
                                                           'HAdobe5k',
                                                           'HFlickr',
                                                           'Hday2night'])
parser.add_argument('--is-cluster', default=False, action='store_true',
                    help="running on TAU cluster or on personal GPU.")

args = parser.parse_args()

IMAGE_HARMONIZATION_DATASET = args.dataset

# IMAGE_HARMONIZATION_DATASET = 'HCOCO'

# is_cluster = False

# if is_cluster:
if args.is_cluster:
    data_root = os.path.join('/storage/jevnisek/ImageHarmonizationDataset/',
                             IMAGE_HARMONIZATION_DATASET)
    target_root = '/storage/jevnisek/ImageHarmonizationResults/mask_prediction/'
else:
    data_root = os.path.join(f'../data/Image_Harmonization_Dataset/',
                             IMAGE_HARMONIZATION_DATASET)
    target_root = os.path.join('mask_detection_results',)

os.makedirs(os.path.join(target_root,
                         IMAGE_HARMONIZATION_DATASET,), exist_ok=True)

test_split = os.path.join(data_root, f'{IMAGE_HARMONIZATION_DATASET}_test.txt')
with open(test_split, 'r') as f:
    test_images = f.read().splitlines()

if args.is_cluster:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_cluster.py')
else:
    config_file = os.path.join('configs',
                               f'{IMAGE_HARMONIZATION_DATASET}_personalGPU.py')
checkpoint_file = os.path.join('checkpoints',
                               f'{IMAGE_HARMONIZATION_DATASET}.pth')
d = torch.load(checkpoint_file)
d['meta']['PALETTE'] = PALETTE
torch.save(d, checkpoint_file)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# test a single image
for test_image_name in tqdm(test_images[:100]):
# test_image_name = test_images[0]

    img = os.path.join(data_root, 'composite_images', test_image_name)
    composite_image = Image.open(img)
    composite_image_tensor = ToTensor()(composite_image).unsqueeze(0)
    mask = os.path.join(data_root, 'masks',
                        convert_composite_image_name_to_mask_name(test_image_name))
    result = inference_segmentor(model, img)

    show_result_pyplot(model, img, result, PALETTE,
                       title='',
                       path=os.path.join(
                           target_root,
                           IMAGE_HARMONIZATION_DATASET,
                           f"{test_image_name}_overlaid_prediction.png"))
    shutil.copy(mask, os.path.join(target_root, IMAGE_HARMONIZATION_DATASET,
                                   f"{test_image_name}_original_mask.png"))
