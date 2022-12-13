import os, json
import matplotlib.pyplot as plt
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default="HCOCO", choices=['HCOCO',
#                                                            'HAdobe5k',
#                                                            'HFlickr',
#                                                            'Hday2night'])
# args = parser.parse_args()

BASE = 'checkpoints/longer_training/jsons'
TARGET_BASE = 'checkpoints/longer_training/jsons/results'
dataset_to_metrics = {}
for DATASET in ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']:
# DATASET = args.dataset


    VAL_EVERY = 20
    os.makedirs(os.path.join(TARGET_BASE, DATASET), exist_ok=True)

    with open(os.path.join(BASE, DATASET+'.json')) as f:
        log_lines = f.read().splitlines()

    new_runs = [pos for pos, l in enumerate(log_lines) if l == "{}"]
    if len(new_runs) > 0:
        log_lines = log_lines[new_runs[-1]+1:]

    train = {'loss': [], 'aux.acc_seg': [], 'iter': []}
    test = {'mIoU': [], 'mAcc': [], 'IoU.original': [], 'IoU.augmented': [],
            'Acc.original': [], 'Acc.augmented': [], 'iter': []}
    for line in log_lines:
        line_data = json.loads(line)
        try:
            if line_data['mode'] == 'train':
                for k in train.keys():
                    train[k].append(line_data[k])
            else:
                for k in test.keys():
                    test[k].append(line_data[k])
        except:
            import ipdb; ipdb.set_trace()

    for k in train.keys():
        plt.clf()
        plt.title(f'{k} @ train')
        plt.plot(train['iter'], train[k])
        plt.xlabel('iteration')
        plt.ylabel(k)
        plt.savefig(os.path.join(TARGET_BASE, DATASET, f"train_{k}.png"))


    for k in test.keys():
        plt.clf()
        plt.title(f'{k} @ test')
        plt.plot(range(VAL_EVERY, (len(test[k]) + 1) * VAL_EVERY, VAL_EVERY), test[k])
        plt.xlabel('iteration')
        plt.ylabel(k)
        plt.savefig(os.path.join(TARGET_BASE, DATASET, f"test_{k}.png"))

    dataset_to_metrics[DATASET] = test


datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']
mIoUs = [dataset_to_metrics[d]['mIoU'][-1] for d in datasets]
plt.clf()
plt.title('mIoU for different datasets')
plt.bar(datasets, mIoUs)
plt.grid(True)
plt.savefig(os.path.join(TARGET_BASE, 'mIoU.png'))

for k in dataset_to_metrics['HCOCO'].keys():
    values = [dataset_to_metrics[d][k][-1] for d in datasets]
    plt.clf()
    plt.title(f'{k} for different datasets')
    plt.bar(datasets, values)
    plt.grid(True)
    plt.savefig(os.path.join(TARGET_BASE, f'{k}.png'))

