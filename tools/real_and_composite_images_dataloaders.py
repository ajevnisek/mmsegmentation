import os
import mmcv
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class FakesAndRealsDataset(Dataset):
    def __init__(self, real_images_paths, fake_images_paths, transform):
        self.real_images_paths = real_images_paths
        self.fake_images_paths = fake_images_paths
        self.transform = transform

    def __getitem__(self, item):
        if item < len(self.real_images_paths):
            image_path = self.real_images_paths[item]
            label = 0
        else:
            image_path = self.fake_images_paths[item - len(self.real_images_paths)]
            label = 1
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        sample = {'path': image_path,
                  'label': label,
                  'image': image_tensor}
        return sample

    def __len__(self):
        return len(self.real_images_paths) + len(self.fake_images_paths)


def create_datasets(data_dir='../data/Image_Harmonization_Dataset',
                    dataset='HAdobe5k', model_name='vgg11'):
    images_paths = get_images_paths_from_filename(data_dir, dataset)
    if model_name == 'vgg11':
        transforms = torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'resnet18':
        transforms = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    else:
        assert False, f"model name {model_name} not supported"

    train_set = FakesAndRealsDataset(
        images_paths['train']['real_images'],
        images_paths['train']['fake_images'],
        transforms
    )
    test_set = FakesAndRealsDataset(
        images_paths['test']['real_images'],
        images_paths['test']['fake_images'],
        transforms
    )
    return train_set, test_set


def create_dataloaders(data_dir='../data/Image_Harmonization_Dataset',
                       dataset='HAdobe5k',
                       model_name='vgg11',
                       train_batch_size=50,
                       test_batch_size=128, num_workers=10):
    train_set, test_set = create_datasets(data_dir, dataset, model_name)
    train_loader = DataLoader(train_set, batch_size=train_batch_size,
                              num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=test_batch_size,
                                 num_workers=num_workers)
    return train_loader, test_dataloader


def convert_fake_name_to_real_name(composite_image_name,
                                   real_image_suffix='.jpg'):
    real_image_filename_without_suffix = composite_image_name.split("_")[0]
    real_image = real_image_filename_without_suffix + real_image_suffix
    return real_image


def get_images_paths_from_filename_for_image_harmonization_datasets(
        base_dir, dataset):
    images_paths = {}
    for mode in ['train', 'test', ]:
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


def get_images_paths_from_filename_for_label_me_datasets(
        base_dir, dataset):
    images_paths = {}
    for mode in ['train', 'test', ]:
        file = os.path.join(base_dir, dataset, f"{dataset}_{mode}.txt")
        all_images = mmcv.list_from_file(file)
        fake_images = [image for image in all_images
                       if image.startswith('composites')]
        real_images = [image for image in all_images
                       if image.startswith('natural_photos')]
        fake_images_paths = [
            os.path.join(base_dir, dataset, x)
            for x in fake_images]
        real_images_paths = [
            os.path.join(base_dir, dataset, x)
            for x in real_images]
        images_paths[mode] = {'real_images': real_images_paths,
                              'fake_images': fake_images_paths}
    return images_paths


def get_images_paths_from_filename(base_dir, dataset):
    if dataset in ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night',]:
        return \
            get_images_paths_from_filename_for_image_harmonization_datasets(
                base_dir, dataset)
    elif dataset in ['LabelMe_all', 'LabelMe_15categories']:
        return get_images_paths_from_filename_for_label_me_datasets(
            base_dir, dataset)

    else:
        assert False, "dataset not supported"
