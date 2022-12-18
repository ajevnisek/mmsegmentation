import os
import argparse

from datetime import datetime

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
    parser.add_argument('--results-dir', type=str, default='results/Hday2night')
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join(args.results_dir,
                               datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    train_loader = get_loader(args.dataset_root, args.dataset,
                              dataset_type='train',
                              batch_size=args.batch_size)
    test_loader = get_loader(args.dataset_root, args.dataset,
                             dataset_type='test')
    discriminator_trainer = GlobalAndLocalFeaturesDiscriminator(train_loader,
                                                                test_loader)
    for epoch in range(1, 1 + args.epochs):
        discriminator_trainer.train_one_epoch()
        results = discriminator_trainer.test_one_epoch()
        curr_metrics = calc_metrics(results)
        generate_graphs(results, curr_metrics, epoch, results_dir)
        log_metrics(curr_metrics, epoch, results_dir)
        print(f"Epoch [{epoch:03d}]: Domain Verification Score: "
              f"{curr_metrics['auc_dove_score'] * 100:.2f} [%]")
        print(f"Epoch [{epoch:03d}]: Global Score: "
              f"{curr_metrics['auc_global_score'] * 100:.2f} [%]")
        print(f"Epoch [{epoch:03d}]: Combined Score: "
              f"{curr_metrics['auc_combined_score'] * 100:.2f} [%]")
        print("~" * 30)


if __name__ == "__main__":
    main()
