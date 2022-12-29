import os
import mmcv

import argparse
from datetime import datetime
from vgg_based_classifier import Trainer


def parse_args():
    parser = argparse.ArgumentParser('Train Real/composite Images classifier '
                                     'based on VGG')
    parser.add_argument('--dataset',
                        choices=['HCOCO', 'HAdobe5k', 'HFlickr',
                                 'Hday2night', 'IHD'],
                        default='Hday2night',
                        help='dataset name.')
    parser.add_argument('--epochs',
                        type=int,
                        default=-1,
                        help='Number of train epochs. Default is -1 which '
                             'means that the number of epochs will be a bit '
                             'higher than 25K iterations.')
    parser.add_argument('--data-dir',
                        default='../data/Image_Harmonization_Dataset/',
                        choices=[
                            '/storage/jevnisek/ImageHarmonizationDataset/',
                            '../data/Image_Harmonization_Dataset/'])
    parser.add_argument('--target-dir', default='results/vgg_training')
    return parser.parse_args()


def convert_fake_name_to_real_name(composite_image_name,
                                   real_image_suffix='.jpg'):
    real_image_filename_without_suffix = composite_image_name.split("_")[0]
    real_image = real_image_filename_without_suffix + real_image_suffix
    return real_image


def get_images_paths_from_filename(base_dir, dataset):
    images_paths = {}
    for mode in ['train', 'test',]:
        file = os.path.join(base_dir, dataset, f"{dataset}_{mode}.txt")
        fake_images = mmcv.list_from_file(file)
        real_images = [convert_fake_name_to_real_name(x) for x in fake_images]
        fake_images_paths = [
            os.path.join(base_dir, dataset, 'composite_images', x)
            for x in fake_images]
        real_images_paths = [
            os.path.join(base_dir, dataset, 'real_images', x)
            for x in real_images]
        images_paths[mode] = {'real_images': real_images_paths,
                              'fake_images': fake_images_paths}
    return images_paths


def main():
    args = parse_args()
    images_paths = get_images_paths_from_filename(args.data_dir, args.dataset)
    target_dir = os.path.join(args.target_dir,
                              datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    os.makedirs(target_dir, exist_ok=True)
    trainer = Trainer(images_paths['train'], images_paths['test'],
                      target_dir)
    trainer.run()


if __name__ == '__main__':
    main()
