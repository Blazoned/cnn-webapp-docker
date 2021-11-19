import os
import random
import sys
from shutil import copy2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataGenerator:
    @property
    def image_size(self):
        return self._image_size

    @property
    def batch_size(self):
        return self._batch_size

    def __init__(self, training_directory: str, validation_directory: str, img_size=48, batch_size: int = 64):
        size_type = type(img_size)
        self.training_directory = training_directory
        self.validation_directory = validation_directory

        self._image_size = img_size if size_type is tuple else \
                           (img_size, img_size) if size_type is int else \
                           (48, 48)
        self._batch_size = batch_size

    def get_generator_for_training(self):
        return self._get_generator()

    def get_generator_for_validation(self):
        return self._get_generator(use_training=False)

    def _get_generator(self, use_training=True):
        img_data_gen = ImageDataGenerator(horizontal_flip=True)
        return img_data_gen.flow_from_directory(self.training_directory if use_training else self.validation_directory,
                                                target_size=self._image_size,
                                                color_mode='grayscale',
                                                batch_size=self._batch_size,
                                                class_mode='categorical',
                                                shuffle=use_training)

    def undersample_data(self, directory: str = ''):
        self._copy_files(*self._prepare_directories(directory, use_oversample_file_count=False))

    def oversample_data(self, directory: str = ''):
        self._copy_files(*self._prepare_directories(directory))

    def _prepare_directories(self, destination: str, use_oversample_file_count: bool = True):
        from_dirs, to_dirs = [], []
        file_count = -sys.maxsize if use_oversample_file_count else sys.maxsize

        for name in os.listdir(self.training_directory):
            file = f'{self.training_directory}/{name}'
            if os.path.isdir(file):
                from_dirs.append(file)

                to_dir = f'{destination}/training/{name}'
                os.makedirs(to_dir)
                file_names = [name for name in os.listdir(from_dirs[-1])
                              if os.path.isfile(f"{from_dirs[-1]}/{name}")]

                sub_count = len(file_names)
                if use_oversample_file_count:
                    if file_count < sub_count:
                        file_count = sub_count
                else:
                    if file_count > sub_count:
                        file_count = sub_count

                to_dirs.append((to_dir, file_names, sub_count))

        return from_dirs, to_dirs, file_count

    @staticmethod
    def _copy_files(from_dirs, to_dirs, file_count):
        for from_dir, to_dir in zip(from_dirs, to_dirs):
            full_copies = file_count // to_dir[2]
            remaining_copies = file_count % to_dir[2]

            for i in range(full_copies+1):
                if i == full_copies:
                    file_list = random.sample(population=to_dir[1], k=remaining_copies)
                else:
                    file_list = to_dir[1]

                for file in file_list:
                    dst_file = f"{file.rstrip('.jpg')}-{i+1}.jpg"
                    copy2(src=f'{from_dir}/{file}', dst=f'{to_dir}/{dst_file}')
