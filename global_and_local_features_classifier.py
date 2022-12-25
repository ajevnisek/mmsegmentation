import os
import argparse

from datetime import datetime

from tools.discriminator_args import DiscriminatorArgs
from tools.discriminator_test_utils import calc_metrics, log_metrics,\
    generate_graphs
from dove_discriminator.discriminator_dataset import get_loader
from dove_discriminator.classifier_based_on_global_and_local_features import \
    GlobalAndLocalFeaturesDiscriminator


def parse_args():
    parser = argparse.ArgumentParser('Discriminator Training.')
    parser.add_argument('--dataset-root', type=str,
                        default='../data/Image_Harmonization_Dataset')
    parser.add_argument('--dataset', type=str, default='Hday2night')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gan-mode', type=str, default='wgangp',
                        choices=['wgangp', 'vanilla', 'lsgan'])
    parser.add_argument('--results-dir', type=str, default='results/Hday2night')
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join(args.results_dir,
                               datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    train_loader = get_loader(args.dataset_root, args.dataset,
                              dataset_type='train',
                              batch_size=args.batch_size, shuffle=True)
    test_loader = get_loader(args.dataset_root, args.dataset,
                             dataset_type='test', shuffle=False)
    discriminator_args = DiscriminatorArgs(learning_rate=args.learning_rate,
                                           gan_mode=args.gan_mode)
    discriminator_trainer = GlobalAndLocalFeaturesDiscriminator(
        train_loader, test_loader, results_dir, discriminator_args)
    discriminator_trainer.run(args.epochs)


if __name__ == "__main__":
    main()
