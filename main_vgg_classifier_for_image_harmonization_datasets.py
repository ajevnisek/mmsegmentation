import os
import json
import mmcv

import argparse
from datetime import datetime
from vgg_based_classifier import Trainer
from tools.real_and_composite_images_dataloaders import get_images_paths_from_filename


LANDONE_DATA_ROOT = '/mnt/data/data/realism_datasets/human_evaluation/'\
                    'lalonde_and_efros_dataset'

def parse_args():
    parser = argparse.ArgumentParser('Train Real/composite Images classifier '
                                     'based on VGG')
    parser.add_argument('--dataset',
                        choices=['HCOCO', 'HAdobe5k', 'HFlickr',
                                 'Hday2night', 'IHD', 'LabelMe_all',
                                 'LabelMe_15categories'],
                        default='Hday2night',
                        help='dataset name.')
    parser.add_argument('--epochs',
                        type=int,
                        default=0,
                        help='Number of train epochs. Default is -1 which '
                             'means that the number of epochs will be a bit '
                             'higher than 25K iterations.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='Batch size, set to labelMe_all default: 50.')
    parser.add_argument('--optimizer-type',
                        type=str,
                        default='SGD',
                        choices=['SGD', 'Adam'],
                        help='Which optimizer, default: SGD.')
    parser.add_argument('--data-dir',
                        default='../data/Image_Harmonization_Dataset/',
                        choices=[
                            '/storage/jevnisek/ImageHarmonizationDataset/',
                            '../data/Image_Harmonization_Dataset/',
                            '../data/realism_datasets/',
                            '/storage/jevnisek/realism_datasets/'
                        ])
    parser.add_argument('--target-dir', default='results/vgg_training')
    parser.add_argument('--landone-root', default=LANDONE_DATA_ROOT,
                        choices=[LANDONE_DATA_ROOT,
                                 '/storage/jevnisek/realism_datasets/'
                                 'human_evaluation/lalonde_and_efros_dataset/'])
    return parser.parse_args()


def main():
    args = parse_args()
    images_paths = get_images_paths_from_filename(args.data_dir, args.dataset)
    target_dir = os.path.join(args.target_dir,
                              datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    trainer = Trainer(images_paths['train'], images_paths['test'],
                      target_dir, epochs=args.epochs,
                      batch_size=args.batch_size,
                      landone_root=args.landone_root,
                      optimizer_type=args.optimizer_type,
                      run_landone_evaluation=args.dataset == 'LabelMe_all')
    trainer.run()


if __name__ == '__main__':
    main()
