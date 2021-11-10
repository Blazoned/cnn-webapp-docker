import os
import random
import sys
from shutil import copy2

import tensorflow.compat.v1 as tf

from traincnn import main as train_cnn


def make_train_directories(to_dirs: tuple = ("copy-train/", "copy-test/"), from_dirs: tuple = ("train/", "test/"),
                           batch_size: int = 16, do_subsample=False):
    # Exit if directories already exist
    if any(os.path.exists(directory) for directory in to_dirs):
        return to_dirs

    # Get subdirectories
    sub_dirs = [[name for name in os.listdir(directory) if os.path.isdir(f"{directory}{name}")]
                for directory in from_dirs]

    # Prepare destination directories
    for directory, sub_directories in zip(to_dirs, sub_dirs):
        for sub_directory in sub_directories:
            os.makedirs(f"{directory}{sub_directory}")

    # Copy files from one directory to another
    files = _subsample_data(to_dirs, from_dirs, sub_dirs, batch_size) if do_subsample else \
            _oversample_data(to_dirs, from_dirs, sub_dirs, batch_size)

    for src_file, dst_file in files:
        copy2(src_file, dst_file)

    return to_dirs


def _subsample_data(to_dirs: tuple, from_dirs: tuple, sub_dirs: list, batch_size: int):
    # Get the maximum amount of items to put in the subdirectories:
    max_lengths = []

    for directory, sub_directories in zip(from_dirs, sub_dirs):
        max_len = sys.maxsize

        for sub_directory in sub_directories:
            sub_dir_len = len([name for name in os.listdir(f"{directory}{sub_directory}")
                               if os.path.isfile(f"{directory}{sub_directory}/{name}")])

            if sub_dir_len == 0:
                return False

            max_len = sub_dir_len if sub_dir_len < max_len else max_len

        max_lengths.append(max_len - (max_len % batch_size))

    # Copy random file from directories until max size for subdirectory is met
    files = []

    for to_directory, from_directory, sub_directories, max_len in zip(to_dirs, from_dirs, sub_dirs, max_lengths):
        for sub_directory in sub_directories:
            files.extend(random.sample(population=[(f"{from_directory}{sub_directory}/{name}",
                                                    f"{to_directory}{sub_directory}/{name}")

                                                   for name in os.listdir(f"{from_directory}{sub_directory}")
                                                   if os.path.isfile(f"{from_directory}{sub_directory}/{name}")],
                                       k=max_len))

        return files


def _oversample_data(to_dirs: tuple, from_dirs: tuple, sub_dirs: list, batch_size: int):
    # Get the minimum amount of items to put in the subdirectories:
    min_lengths = []

    for directory, sub_directories in zip(from_dirs, sub_dirs):
        min_len = -sys.maxsize

        for sub_directory in sub_directories:
            sub_dir_len = len([name for name in os.listdir(f"{directory}{sub_directory}")
                               if os.path.isfile(f"{directory}{sub_directory}/{name}")])

            if sub_dir_len == 0:
                return False

            min_len = sub_dir_len if sub_dir_len > min_len else min_len

        min_lengths.append(min_len - (min_len % batch_size))

    # Copy random file from directories until max size for subdirectory is met
    files = []

    for to_directory, from_directory, sub_directories, min_len in zip(to_dirs, from_dirs, sub_dirs, min_lengths):
        for sub_directory in sub_directories:
            sub_dir_len = len([name for name in os.listdir(f"{from_directory}{sub_directory}")
                               if os.path.isfile(f"{from_directory}{sub_directory}/{name}")])

            full_copies = min_len // sub_dir_len
            remaining_copies = min_len % sub_dir_len

            files_names = [name.replace(".jpg", "") for name in os.listdir(f"{from_directory}{sub_directory}")
                           if os.path.isfile(f"{from_directory}{sub_directory}/{name}")]

            files.extend([(f"{from_directory}{sub_directory}/{name}.jpg",
                           f"{to_directory}{sub_directory}/{name}-{i}.jpg")

                          for name in files_names
                          for i in range(full_copies)])

            files.extend(random.sample(population=[(f"{from_directory}{sub_directory}/{name}.jpg",
                                                    f"{to_directory}{sub_directory}/{name}-{full_copies}.jpg")
                                                   for name in files_names],
                                       k=remaining_copies))

    return files


def train_model(directories: tuple = ("train/", "test/"), batch_size: int = 16, category_name: str = ""):
    tf.enable_eager_execution()
    tf.disable_v2_behavior()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4096
    config.inter_op_parallelism_threads = 4096
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device("/DML:0"):
        train_cnn(directories=directories, batch_size=batch_size, category_name=category_name)


def main(to_dirs: tuple = ("train-gpu/", "test-gpu/"), from_dirs: tuple = ("train/", "test/"), batch_size: int = 16):
    to_dirs = ("oversampled-big-batch-train/", "oversampled-big-batch-test/")
    batch_size = 256
    directories = make_train_directories(to_dirs, from_dirs, batch_size)
    train_model(directories, batch_size, "oversampled-big-batch-no-reduce-learning")


if __name__ == '__main__':
    main()
