import os
import json
import logging

import torch

import matplotlib.pyplot as plt

from sklearn import metrics


def calc_metrics(results_dict):
    dove_score = torch.cat([
        results_dict['composite_image_domain_similarity_score'],
        results_dict['real_image_domain_similarity_score']])
    gt_labels = torch.cat([
        torch.ones_like(
            results_dict['composite_image_domain_similarity_score']),
        torch.zeros_like(results_dict['real_image_domain_similarity_score'])])
    auc_dove_score = metrics.roc_auc_score(gt_labels, dove_score)
    auc_dove_score = 1 - auc_dove_score if auc_dove_score < 0.5 else \
        auc_dove_score
    global_score = torch.cat([
        results_dict['composite_image_global_score'],
        results_dict['real_image_global_score']])
    gt_labels = torch.cat([
        torch.ones_like(results_dict['composite_image_global_score']),
        torch.zeros_like(results_dict['real_image_global_score'])])
    auc_global_score = metrics.roc_auc_score(gt_labels, global_score)
    auc_dove_score = 1 - auc_global_score if auc_global_score < 0.5 else \
        auc_global_score
    combined_score = dove_score + global_score
    auc_combined_score = metrics.roc_auc_score(gt_labels, combined_score)
    auc_combined_score = 1 - auc_combined_score if auc_combined_score < 0.5 \
        else auc_combined_score
    return {"auc_dove_score": auc_dove_score,
            "auc_global_score": auc_global_score,
            "auc_combined_score": auc_combined_score}


def log_metrics(curr_metrics, epoch, results_dir):
    epoch_results_dir = os.path.join(results_dir, f'epoch_{epoch:03d}')
    with open(os.path.join(epoch_results_dir, 'metrics.log'), 'w') as f:
        json.dump(curr_metrics, f, indent=2)


def generate_graphs(results_dict, curr_metrics, epoch, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    epoch_results_dir = os.path.join(results_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_results_dir, exist_ok=True)
    plt.clf()
    plt.title(f"Domain Verification Score, "
              f"AuC:{curr_metrics['auc_dove_score'] * 100.0:.2f}")
    plt.grid(True)
    plt.hist(results_dict['composite_image_domain_similarity_score'], alpha=0.5,
             label='composite')
    plt.hist(results_dict['real_image_domain_similarity_score'], alpha=0.5,
             label='real')
    plt.legend()
    plt.savefig(os.path.join(epoch_results_dir, 'dove_score.png'))

    plt.clf()
    plt.title(f"Global Score"
              f"AuC:{curr_metrics['auc_global_score'] * 100.0:.2f}")
    plt.grid(True)
    plt.hist(results_dict['composite_image_global_score'], alpha=0.5,
             label='composite')
    plt.hist(results_dict['real_image_global_score'], alpha=0.5,
             label='real')
    plt.legend()
    plt.savefig(os.path.join(epoch_results_dir, 'global_score.png'))

    plt.clf()
    plt.title(f"Combined Score"
              f"AuC:{curr_metrics['auc_combined_score'] * 100.0:.2f}")
    plt.grid(True)
    plt.hist(results_dict['composite_image_global_score'] +
             results_dict['composite_image_domain_similarity_score'],
             alpha=0.5, label='composite')
    plt.hist(results_dict['real_image_global_score'] +
             results_dict['real_image_domain_similarity_score'],
             alpha=0.5,
             label='real')
    plt.legend()
    plt.savefig(os.path.join(epoch_results_dir, 'combined_score.png'))


def get_logger(logger_file):
    # create logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_file)
    fh.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # add formatter to ch
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(fh)
    return logger
