from .model import ConvolutionalModelGenerator
from .data_generator import DataGenerator

import os

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt


def train_cnn(model_generator: ConvolutionalModelGenerator, generator: DataGenerator, epochs: int = 500,
                model_directory: str = '', do_reduce_learning: bool = False, do_checkpoints: bool = False,
                early_stopping_delta: float = 0.001, early_stopping_patience: int = 0):
    training_generator = generator.get_generator_for_training()
    validation_generator = generator.get_generator_for_validation()

    training_steps = training_generator.n // training_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size

    if not model_generator.is_compiled:
        model_generator.compile()

    model = model_generator.model

    callbacks = []
    if do_reduce_learning:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss'))
    if do_checkpoints:
        callbacks.append(ModelCheckpoint(f"{model_directory}{'/model_weights.{epoch:04d}-{val_acc:.3f}.hdf5'}",
                                         monitor='val_acc', save_weights_only=True))
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor='val_acc', min_delta=early_stopping_delta,
                                       patience=early_stopping_patience))

    os.makedirs(model_directory, exist_ok=True)
    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    if do_checkpoints:
        __plot_history(history)
    else:
        model.save_weights(f'{model_directory}/model_weights.hdf5', save_format='h5')

    with open(f'{model_directory}/model.json', 'w') as f:
        f.write(model.to_json())


def __plot_history(history):
    # Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
