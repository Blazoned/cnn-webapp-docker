import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from livelossplot import PlotLossesKerasTF
from matplotlib import pyplot as plt
import tensorflow as tf

print("Tensorflow version:", tf.__version__)


def print_expressions():
    for expression in os.listdir("train/"):
        print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")


def get_data_generators(directories: tuple = ("train/", "test/"), img_size=48, batch_size: int = 64):
    size_type = type(img_size)
    img_size = img_size if size_type is tuple else \
               (img_size, img_size) if size_type is int else \
               (48, 48)

    datagen_train = ImageDataGenerator(horizontal_flip=True)

    train_generator = datagen_train.flow_from_directory(directories[0],
                                                        target_size=img_size,
                                                        color_mode="grayscale",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    datagen_validation = ImageDataGenerator(horizontal_flip=True)
    validation_generator = datagen_validation.flow_from_directory(directories[1],
                                                                  target_size=img_size,
                                                                  color_mode="grayscale",
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=False)

    return train_generator, validation_generator


def build_model():
    # Initialising the CNN
    model = Sequential()

    def add_convolutional_layer(filters, kernel_size, padding, **kwargs):
        model.add(Conv2D(filters, kernel_size, padding=padding, **kwargs))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    def add_dense_layer(units):
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    # 1 - Convolution layers x4
    add_convolutional_layer(64, (3, 3), padding='same', input_shape=(48, 48, 1))
    add_convolutional_layer(128, (5, 5), padding='same')
    add_convolutional_layer(512, (3, 3), padding='same')
    add_convolutional_layer(512, (3, 3), padding='same')

    # 2 - Flattening
    model.add(Flatten())

    # 3 - Fully connected layers x2
    add_dense_layer(256)
    add_dense_layer(512)

    # 4 - Output layer
    model.add(Dense(7, activation='softmax'))

    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def train_model(directories: tuple = ("train/", "test/"), img_size=48, batch_size: int = 64, category_name: str = ""):
    start_train = input("Start training? (Y/n)")

    if start_train == "Y":
        train_generator, validation_generator = get_data_generators(directories, img_size, batch_size)
        model = build_model()

        epochs = 5000
        steps_per_epoch = train_generator.n // train_generator.batch_size
        validation_steps = validation_generator.n // validation_generator.batch_size
        category_name = f"{category_name}-" if category_name is not "" else category_name

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                               patience=2, min_lr=0.00001, mode='auto')
        checkpoint = ModelCheckpoint(f"{category_name}{'weights/model_weights.{epoch:04d}-{val_acc:.2f}.hdf5'}",
                                     monitor='val_acc', save_weights_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, min_delta=0.001, patience=10)
        callbacks = [early_stopping, checkpoint]

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        # Plot accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    if start_train == "n" or start_train == "Y":
        return False
    else:
        return True


def main(directories: tuple = ("train/", "test/"), img_size=48, batch_size: int = 64, category_name: str = ""):
    while train_model(directories, img_size, batch_size, category_name):
        pass


if __name__ == '__main__':
    main()
