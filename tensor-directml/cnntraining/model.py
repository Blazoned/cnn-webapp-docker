from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class ConvolutionalModelGenerator:
    _model: Sequential

    _learning_rate: float
    _convolutional_layers: list
    _dense_layers: list

    _is_compiled: bool

    # region Properties
    @property
    def model(self):
        return self._model

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def convolutional_layers(self):
        return self._convolutional_layers

    @property
    def dense_layers(self):
        return self._dense_layers

    @property
    def is_compiled(self):
        return self._is_compiled
    # endregion

    def __init__(self):
        self._categories = []
        self._convolutional_layers = []
        self._dense_layers = []
        self._is_compiled = False

    # region Convolutional Layer
    def add_convolutional_layer(self, filter_count: int, kernel_size: tuple, padding: str = 'same',
                                activation: str = 'relu', dropout: float = 0.0):
        self._convolutional_layers.append(self._write_convolutional_layer(filter_count, kernel_size, padding,
                                                                          activation, dropout))

    def insert_convolutional_layer(self, index: int, filter_count: int, kernel_size: tuple, padding: str = 'same',
                                   activation: str = 'relu', dropout: float = 0.0):
        if index < len(self._convolutional_layers):
            self._convolutional_layers.insert(index,
                                              self._write_convolutional_layer(filter_count, kernel_size, padding,
                                                                              activation, dropout))

    def remove_convolutional_layer(self, index: int):
        if index < len(self._convolutional_layers):
            del self._convolutional_layers[index]

    def _write_convolutional_layer(self, filter_count: int, kernel_size: tuple, padding: str,
                                   activation: str, dropout: float):
        self._is_compiled = False
        return {
            'filter_count': filter_count,
            'kernel_size': kernel_size,
            'padding': padding,
            'activation': activation,
            'dropout': dropout
        }
    # endregion

    # region Dense Layer
    def add_dense_layer(self, nodes: int, activation: str = 'relu', dropout: float = 0.0):
        self._dense_layers.append(self._write_dense_layer(nodes, activation, dropout))

    def insert_dense_layer(self, index: int, nodes: int, activation: str = 'relu', dropout: float = 0.0):
        if index < len(self._dense_layers):
            self._dense_layers.insert(index, self._write_dense_layer(nodes, activation, dropout))

    def remove_dense_layer(self, index: int):
        if index < len(self._dense_layers):
            del self._dense_layers[index]

    def _write_dense_layer(self, nodes: int, activation: str, dropout: float):
        self._is_compiled = False
        return {
            'node_count': nodes,
            'activation': activation,
            'dropout': dropout
        }
    # endregion

    # region Compilation
    def compile(self, learning_rate: float = 0.0005, do_summary=False):
        c_layer_count = len(self._convolutional_layers)

        if c_layer_count == 0:
            raise LookupError('You must define convolutional before compiling.')

        self._model = Sequential()

        self._compile_convolutional_layer(self._convolutional_layers[0])

        if c_layer_count >= 2:
            for c_layer in self._convolutional_layers[1:]:
                self._compile_convolutional_layer(c_layer)

        self._model.add(Flatten())

        for d_layer in self._dense_layers:
            self._compile_dense_layer(d_layer)

        self._compile_dense_layer(self._write_dense_layer(7, 'softmax', 0.0))

        self._learning_rate = learning_rate
        optimiser = Adam(learning_rate=self._learning_rate)
        self._model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

        if do_summary:
            self._model.summary()

        return self._model

    def _compile_convolutional_layer(self, c_layer: dict, is_first: bool = False):
        if is_first:
            self._model.add(Conv2D(c_layer['filter_count'], c_layer['kernel_size'], padding=c_layer['padding'],
                                   input_shape=(48, 48, 1)))
        else:
            self._model.add(Conv2D(c_layer['filter_count'], c_layer['kernel_size'], padding=c_layer['padding']))

        self._model.add(BatchNormalization())
        self._model.add(Activation(c_layer['activation']))
        self._model.add(MaxPooling2D(pool_size=(2, 2)))
        self._model.add(Dropout(c_layer['dropout']))

    def _compile_dense_layer(self, d_layer: dict):
        self._model.add(Dense(d_layer['node_count']))
        self._model.add(BatchNormalization())
        self._model.add(Activation(d_layer['activation']))
        self._model.add(Dropout(d_layer['dropout']))
    # endregion
