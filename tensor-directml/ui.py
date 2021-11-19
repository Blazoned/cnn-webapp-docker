import os.path
from time import sleep

from cnntraining.training import train_cnn
from cnntraining.model import ConvolutionalModelGenerator
from cnntraining.data_generator import DataGenerator

import tensorflow.compat.v1 as tf


def __settings(key: str = '', value=None, settings: dict = {}):
    settings['train_dir']: str = './train'
    settings['val_dir']: str = './test'
    settings['sampling']: str = 'undersampling'
    settings['img_size'] = 48
    settings['batch_size']: int = 64

    if value:
        settings.setdefault(key, value)

    return settings.get(key, settings)


def __init_keras():
    tf.enable_eager_execution()
    tf.disable_v2_behavior()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4096
    config.inter_op_parallelism_threads = 4096
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # TODO: put code piece where the model is supposed to be trained
    with tf.device("/DML:0"):
        pass  # TODO: invoke training here


def __display_introduction():
    # TODO: add introduction message
    __display_main_menu()


def __display_main_menu():
    print("""
[CNN Model Trainer - Menu]

[I] Manage image data settings
[M] Manage model settings
[S] Manage training settings
[T] Train model

[Q] Quit
""")

    response = []
    while not __get_input_within_range(allowed_input=['I', 'M', 'S', 'T'], result=response):
        pass

    response = response[0]

    if response == 'I':
        __display_data_settings()
    elif response == 'M':
        __display_model_settings()
    elif response == 'S':
        __display_training_settings()
    elif response == 'T':
        __display_train_model()
    elif response == 'Q':
        __display_quit()


def __display_data_settings():
    print(f"""
[CNN Model Trainer - Image Data Settings]

Current settings:
{__get_settings_message('images')}

[T] Training directory
[V] Validation directory
[S] Sampling mode
[I] Image size
[B] Batch size

[Q] Back
""")

    response = []
    while not __get_input_within_range(allowed_input=['T', 'V', 'I', 'B', 'Q'], result=response):
        pass

    response = response[0]

    if response == 'T':  # Training directory
        msg = f"""
The training directory defines the location of the training data (e.g. ./train). The directory should have the same 
subdirectories as the validation directory. Each subdirectory defines an emotion (category). You cannot put an image 
directly into the directory itself, and the subdirectories need to have at least a single image (.jpg only).

New training directory: 
"""
        print(msg)
        response = ''

        while not response:
            response = input()

            if os.path.exists(response) and any([os.path.isdir(item)
                                                 and any([os.path.isfile(file)
                                                          and file.lower().endswith('.jpg')
                                                          for file in os.listdir(f'response/{item}')])
                                                 for item in os.listdir(response)]):
                __settings(key='train_dir', value=int(response))
            else:
                response = ''
                print(f"""
Invalid path '{response}'!
""")
                sleep(secs=1.5)
                print(msg)

        print("""
Successfully updated the training directory.""")
    elif response == 'V':  # Validation directory
        msg = f"""
The validation directory defines the location of the validation data (e.g. ./test). The directory should have the same 
subdirectories as the training directory. Each subdirectory defines an emotion (category). You cannot put an image 
directly into the directory itself, and the subdirectories need to have at least a single image (.jpg only).

New validation directory: 
"""
        print(msg)
        response = ''

        while not response:
            response = input()

            if os.path.exists(response) and any([os.path.isdir(item)
                                                 and any([os.path.isfile(file)
                                                          and file.lower().endswith('.jpg')
                                                          for file in os.listdir(f'response/{item}')])
                                                 for item in os.listdir(response)]):
                __settings(key='val_dir', value=int(response))
            else:
                response = ''
                print(f"""
        Invalid path '{response}'!
        """)
                sleep(secs=1.5)
                print(msg)

        print("""
Successfully updated the validation directory.""")

    elif response == 'S':  # SAMPLING
        msg = f"""
Sampling of the dataset is used for balancing the amount of images available for each emotion (category). This can 
either be 'undersampling', 'oversampling' or 'none'.
Undersampling means the emotion with the least amount of images will be used to determine how many images the other 
emotions have available to them.
Oversampling means instead of the the emotion with the least amount of images, the emotion with the most amount of 
images will be used as a baseline and other emotions will clone images until the right amount of images are available.
None means the datasets will not be balanced.

New batch size: 
"""
        print(msg)

        response = []
        allowed_input = ['none', 'undersampling', 'oversampling']
        while not __get_input_within_range(allowed_input=allowed_input, result=response):
            response = ''
            print(f"""
Invalid value '{response}'! The only allowed values are {allowed_input}.
""")
            sleep(secs=2.5)
            print(msg)

        response = response[0]

        print("""
        Successfully updated the sampling mode.""")

    elif response == 'I':  # IMAGE SIZE
        msg = f"""
Images size defines the size of all images in the dataset. All images need to be the same size. You can define both the 
width and height like so '<width>x<height>' (e.g. 128x384) or you can enter a single integer (e.g. 48) if the size of 
the images is square.

New image size: 
"""
        print(msg)
        response = ''

        while not response:
            response = input()

            if response.isnumeric():
                __settings(key='img_size', value=int(response))
            elif ('x' in response) and (response.replace('x', '', __count=1).isnumeric()):
                width, height = response.split('x')
                __settings(key='img_size', value=(int(width), int(height)))
            else:
                response = ''
                print(f"""
Invalid value '{response}'!
""")
                sleep(secs=1.5)
                print(msg)

        print("""
Successfully updated the image size.""")

    elif response == 'B':  # BATCH SIZE
        msg = f"""
Batch size of the dataset defines how many images are being bundled together for processing. This needs to be a positive 
integer value (e.g. 64). It is recommended to use a power of 2 below the value of 128. Other values might pose problems 
during the training process.

New batch size: 
"""
        print(msg)
        response = ''

        while not response:
            response = input()

            if response.isnumeric():
                __settings(key='batch_size', value=int(response))
            else:
                response = ''
                print(f"""
Invalid value '{response}'!
""")
                sleep(secs=1.5)
                print(msg)

        print("""
Successfully updated the batch size.""")

    if response == 'Q':
        __display_main_menu()
    else:
        __display_data_settings()


def __display_model_settings():
    pass


def __display_training_settings():
    pass


def __display_train_model():
    pass


def __display_quit():
    pass


def __get_input_within_range(allowed_input: list, case_sensitive: bool = False, result: list = []):
    result[0] = input()

    if not case_sensitive:
        allowed_input = [item.lower() for item in allowed_input]

    if result[0] not in allowed_input:
        result[0] = ''
        print(f"""
Invalid option '{result[0]}'. The response needs to be one of the following options: {result[0]}.
""")
        sleep(secs=2.5)

    return result[0]


def __get_settings_message(segment: str = ''):
    msg = ''

    if segment == 'images' or segment == '':
        msg += f"""
\ttraining directory: {__settings(key='train_dir')}
\tvalidation directory: {__settings(key='val_dir')}
\tsampling: {__settings(key='sampling')}
\timage size (expected):{__settings(key='img_size')}
\tbatch size: {__settings(key='batch_size')}
"""

    if segment == 'model' or segment == '':
        pass

    return msg


def start():
    __init_keras()
    __display_introduction()


if __name__ == '__main__':
    start()
