import shutil

from os import listdir
from os.path import isfile, join


dir_dataset_src = 'dataset-raw'
dir_images_src = f'{dir_dataset_src}/images'
dir_labels_src = f'{dir_dataset_src}/labels'

dir_dataset_dst = 'dataset'
dir_with_pool_dst = f'{dir_dataset_dst}/com-piscina'
dir_without_pool_dst = f'{dir_dataset_dst}/sem-piscina'


def load_images(dir_images):
    images = [f for f in listdir(dir_images) if isfile(join(dir_images, f))]
    print(f'Images count = {len(images)}')
    print(f'Images = {images}')
    return images


def load_labels(dir_labels):
    label_names = [f.title().split('.')[0] for f in listdir(dir_labels) if isfile(join(dir_labels, f))]
    print(f'Labels count = {len(label_names)}')
    print(f'Labels = {label_names}')
    return label_names


def remove_ext_from_image_name(image):
    image_name = image.title()
    return image_name.split('.')[0]


def copy_image_to_dataset_dir(image, dir_dst):
    image_name_src = f'{dir_images_src}/{image}'
    image_name_dst = f'{dir_dst}/{image.title().split(".")[0].zfill(3)}.png'
    shutil.copy(image_name_src, image_name_dst)


images = load_images(dir_images_src)
labels = load_labels(dir_labels_src)

for image in images:
    image_name_without_ext = remove_ext_from_image_name(image)

    if image_name_without_ext in labels:
        copy_image_to_dataset_dir(image, dir_with_pool_dst)
    else:
        copy_image_to_dataset_dir(image, dir_without_pool_dst)
