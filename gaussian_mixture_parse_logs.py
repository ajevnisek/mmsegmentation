import numpy as np
import matplotlib.pyplot as plt


datasets = [
    'HAdobe5k',
    'HCOCO',
    'Hday2night',
    'HFlickr'
]
for dataset in datasets:
    with open(f'aucs_{dataset}.json', 'r') as f:
        data = f.read()
    lines = data.splitlines()
    lines = [l for l in lines if "auc" in l]
    aucs = [float(l.split('"')[-2]) for l in lines]

    plt.clf()
    plt.title(f'Per pixel AuC histogram for {dataset} dataset')
    plt.hist(aucs, bins=int(np.sqrt(len(aucs))))
    plt.grid(True)
    plt.xlabel('AuC')
    plt.ylabel('count')
    plt.savefig(f'gaussian_mixture_auc_histogram_{dataset}.png')
