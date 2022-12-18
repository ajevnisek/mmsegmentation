import os
import mmcv
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader


DATASET_ROOT = '../data/Image_Harmonization_Dataset/'
DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

DEFAULT_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class ImageHarmonizationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, dataset_name, dataset_type='train',
                 image_transform=None, mask_transform=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        text_file_path = os.path.join(dataset_root, dataset_name,
                                      f"{dataset_name}_{dataset_type}.txt")
        self.composite_images_names = mmcv.list_from_file(text_file_path)
        self.composite_images = [
            os.path.join(dataset_root, dataset_name, 'composite_images', name)
            for name in self.composite_images_names]
        self.real_images = [
            os.path.join(dataset_root, dataset_name, 'real_images',
                         self.composite_name_to_real_name(name))
            for name in self.composite_images_names]
        self.masks = [
            os.path.join(dataset_root, dataset_name, 'masks',
                         self.composite_name_to_mask_name(name, '.png'))
            for name in self.composite_images_names]

    @staticmethod
    def composite_name_to_mask_name(composite_image_name,
                                    seg_map_suffix='.png'):
        seg_map_filename_without_suffix = '_'.join(
            composite_image_name.split("_")[:2])
        seg_map = seg_map_filename_without_suffix + seg_map_suffix
        return seg_map

    @staticmethod
    def composite_name_to_real_name(composite_image_name,
                                    real_image_suffix='.jpg'):
        real_image_filename_without_suffix = composite_image_name.split("_")[0]
        real_image = real_image_filename_without_suffix + real_image_suffix
        return real_image

    def __getitem__(self, item):
        real_image_path = self.real_images[item]
        composite_image_path = self.composite_images[item]
        mask_path = self.masks[item]

        real_image = Image.open(real_image_path).convert('RGB')
        composite_image = Image.open(composite_image_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        if self.image_transform:
            real_image = self.image_transform(real_image)
            composite_image = self.image_transform(composite_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {'real_image': real_image,
                  'composite_image': composite_image,
                  'mask': mask,
                  'real_image_path': real_image_path,
                  'composite_image_path': composite_image_path,
                  'mask_path': mask_path}
        return sample

    def __len__(self):
        return len(self.real_images)


def get_dataset(dataset_root, dataset_name, dataset_type='train'):
    dataset = ImageHarmonizationDataset(dataset_root=dataset_root,
                                        dataset_name=dataset_name,
                                        dataset_type=dataset_type,
                                        image_transform=DEFAULT_IMAGE_TRANSFORM,
                                        mask_transform=DEFAULT_MASK_TRANSFORM)
    return dataset


def get_loader(dataset_root, dataset_name, dataset_type='train', batch_size=128,
               shuffle=True):
    dataloader = DataLoader(get_dataset(dataset_root, dataset_name,
                                        dataset_type=dataset_type),
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader
