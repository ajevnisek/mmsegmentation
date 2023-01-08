import os
import mmcv


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
