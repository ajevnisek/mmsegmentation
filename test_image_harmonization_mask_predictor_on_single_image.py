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
PALETTE = [[128, 128, 128], [147,20,255],]
GT_PALETTE = [[128, 128, 128], [144,238,144],]



def custom_show_result(img,
                       result,
                       palette=GT_PALETTE,
                       win_name='',
                       show=False,
                       wait_time=0,
                       out_file=None,
                       opacity=0.5,
                       classes=CLASSES):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]

    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.int64)
    color_seg = img.astype(np.int64)
    for label, color in enumerate(palette):
        if label == 1:
            color_seg[seg == label, :] = color

    # convert to BGR
    # color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img

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
parser.add_argument('--checkpoints-dir', type=str, default="checkpoints",
                    help='checkpoints root directory [checkpoints, '
                         'checkpoints/longer_training]')
parser.add_argument('--target-dir', type=str, default="presentation",
                    help='root directory for saving real, composite and '
                         'overlaid images.')

args = parser.parse_args()

# IMAGE_HARMONIZATION_DATASET = 'HCOCO'
# IMAGE_HARMONIZATION_DATASET = 'HAdobe5k'
# IMAGE_HARMONIZATION_DATASET = 'HFlickr'
IMAGE_HARMONIZATION_DATASET = args.dataset

is_cluster = False
data_root = os.path.join(f'../data/Image_Harmonization_Dataset/',
                         IMAGE_HARMONIZATION_DATASET)

test_split = os.path.join(data_root, f'{IMAGE_HARMONIZATION_DATASET}_test.txt')
with open(test_split, 'r') as f:
    test_images = f.read().splitlines()
config_file = os.path.join('configs',
                           f'{IMAGE_HARMONIZATION_DATASET}_personalGPU.py')
checkpoint_file = os.path.join(args.checkpoints_dir,
                               f'{IMAGE_HARMONIZATION_DATASET}.pth')

d = torch.load(checkpoint_file)
d['meta']['PALETTE'] = PALETTE
torch.save(d, checkpoint_file)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
test_image_name = 'c10580_600936_1.jpg'
test_images.sort()
test_images = test_images[:100]
for test_image_name in test_images:
    # dirname = 'giraffe'
    dirname = os.path.join(IMAGE_HARMONIZATION_DATASET, test_image_name[:-4])
    img = os.path.join(data_root, 'composite_images', test_image_name)
    composite_image = Image.open(img)
    real_img = os.path.join(data_root, 'real_images',
                            convert_composite_image_name_to_real_image_name(
                                test_image_name))
    real_image = Image.open(real_img)

    result = inference_segmentor(model, img)
    overlaid_img = custom_show_result(img, result,
                                     palette=PALETTE, show=False, opacity=0.5)

    overlaid_image_rgb = mmcv.bgr2rgb(overlaid_img)
    overlaid_pil_image_rgb = Image.fromarray(np.uint8(
        overlaid_image_rgb)).convert('RGB')
    mask = os.path.join(
        data_root, 'masks', convert_composite_image_name_to_mask_name(
            test_image_name))
    overlaid_ground_truth = custom_show_result(
        img, (ToTensor()(Image.open(mask))> 0.5).float(),
                                                  palette=GT_PALETTE, show=False,
                                                  opacity=0.5)
    os.makedirs(os.path.join(args.target_dir, dirname,),
                exist_ok=True)
    real_image.save(os.path.join(args.target_dir, dirname, 'real.png'))
    composite_image.save(
        os.path.join(args.target_dir, dirname, 'composite.png'))
    overlaid_pil_image_rgb.save(
        os.path.join(args.target_dir, dirname, 'detection_on_composite.png'))
    Image.fromarray(mmcv.bgr2rgb(overlaid_ground_truth)).convert('RGB').save(
        os.path.join(args.target_dir, dirname,'ground_truth.png'))

    harmonized_by_harmonizer_data_root = os.path.join(
            f'../Harmonizer/dataset/harmonized_images/',
            IMAGE_HARMONIZATION_DATASET)
    harmonized_img_path = os.path.join(harmonized_by_harmonizer_data_root,
                                       test_image_name)
    harmonized_by_harmonizer = Image.open(harmonized_img_path)
    harmonized_by_harmonizer.save(
        os.path.join(args.target_dir, dirname, 'harmonized_image.png'))
    harmonized_by_harmonizer_tensor = ToTensor()(
        harmonized_by_harmonizer).unsqueeze(0)
    result_on_harmonized = inference_segmentor(model, harmonized_img_path)
    detected_harmonized_overlaid_img = custom_show_result(
        harmonized_img_path, result_on_harmonized, palette=PALETTE,  show=False,
        opacity=0.5)
    detected_harmonized_overlaid_img_rgb = mmcv.bgr2rgb(
        detected_harmonized_overlaid_img)
    detected_harmonized_overlaid_pil_image_rgb = Image.fromarray(np.uint8(
        detected_harmonized_overlaid_img_rgb)).convert('RGB')
    detected_harmonized_overlaid_pil_image_rgb.save(
        os.path.join(args.target_dir, dirname, 'detected_on_harmonized.png'))

    result_on_real = inference_segmentor(model, real_img)
    detected_real_overlaid_img = custom_show_result(
        real_img, result_on_real, palette=PALETTE,  show=False,
        opacity=0.5)
    detected_real_overlaid_img_rgb = mmcv.bgr2rgb(
        detected_real_overlaid_img)
    detected_real_overlaid_pil_image_rgb = Image.fromarray(np.uint8(
        detected_real_overlaid_img_rgb)).convert('RGB')
    detected_real_overlaid_pil_image_rgb.save(
        os.path.join(args.target_dir, dirname, 'detected_on_real.png'))

