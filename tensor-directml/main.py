from ui import start

from traincnn import main as train_cnn


def train_model(directories: tuple = ("train/", "test/"), batch_size: int = 16, category_name: str = ""):
    with tf.device("/DML:0"):
        train_cnn(directories=directories, batch_size=batch_size, category_name=category_name)


def main(to_dirs: tuple = ("train-gpu/", "test-gpu/"), from_dirs: tuple = ("train/", "test/"), batch_size: int = 16):
    to_dirs = ("oversampled-big-batch-train/", "oversampled-big-batch-test/")
    batch_size = 256
    # directories = make_train_directories(to_dirs, from_dirs, batch_size)
    # train_model(directories, batch_size, "oversampled-big-batch-no-reduce-learning")


if __name__ == '__main__':
    start()
