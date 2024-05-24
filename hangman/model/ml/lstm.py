"""Define an LSTM model to guess letters"""

import os
from typing import Type, List

import numpy as np
import tensorflow.keras.callbacks
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.models
import tensorflow.keras.preprocessing.sequence
import tensorflow.keras.utils

import hangman.core.dictionary
import hangman.model.ml.utils
from hangman.model.ml.config.iconfig import IConfig
from hangman.model.ml.imodel import IModel

OUTPUT_PATH = os.path.join(hangman.core.dictionary.DATA)
SEQUENCE_LENGTH = 5


class LSTModel(IModel):
    """
    Defines a container for creating a LSTM model with
    functionality to train it & use it for predictions
    """

    def __init__(
            self,
            build_or_load: str,
            *,
            config: IConfig,
            ouput_path: str = OUTPUT_PATH,
            pad_sequence: bool = True,
            sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        """
        Create LSTM model container

        :param build_or_load: (str) build or load the model from file. Options:
            - 'build' - build in-memory 'x' & 'y' cannot be None
            - 'build_weights' - same as 'build' but also loads weights from file
                                'weights_path' cannot be None
            - 'load_model_weights' - loads model & weights from files. 'model_path'
                                     & 'weights_path' cannot be None
        :param config: (IConfig) instance to build & load LSTM model
        :param ouput_path: (str) directory to output weights to during training
        :param pad_sequence: (bool) whether to pad input data for prediction
        :param sequence_length: (int) input sequence length
        """

        # Store instance params
        self.config = config
        self.ouput_path = ouput_path
        self.pad_sequence = pad_sequence
        self.sequence_length = sequence_length

        # Load model
        _model = None
        build_or_load = build_or_load.lower()
        if build_or_load == "build":
            _model = self.config.build()

        elif build_or_load == "build_weights":
            _model = self.config.build()
            _model = self.config.load_weights(model=_model, compile_model=False)

        elif build_or_load == "load_model":
            _model = self.config.load(compile_model=True)

        elif build_or_load == "load_model_weights":
            _model = self.config.load(compile_model=False)
            _model = self.config.load_weights(model=_model, compile_model=True)

        else:
            raise ValueError(f"Invalid value for build_or_load: [{build_or_load}]")

        self.__model = _model

        self.ouput_path = ouput_path

    def train(self, epochs: int = 50, batch_size: int = 64):
        """Train model"""

        return self.__model.fit(
            self.x,
            self.y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=call_backs(self.ouput_path) if self.ouput_path is not None else None
        )

    def predict(self, x: Type["np.array"]) -> str:
        """
        Predict y values based on pre-trained model

        :param x: (np.array) 3D array (# Samples, # Time Steps, # Features)
        :return: (str) prediction
        """

        p = [hangman.model.ml.utils.TO_INT[p] for p in x]

        _sequence_length = len(x)
        if self.pad_sequence:
            p = tensorflow.keras.preprocessing.sequence.pad_sequences(
                [p], padding="post", maxlen=self.sequence_length
            )
            _sequence_length = self.sequence_length

        p = np.array(p).reshape(1, _sequence_length, 1)
        p = p / len(hangman.model.ml.utils.TO_CHAR)

        prediction = self.__model.predict(p, verbose=0)
        index = np.argmax(prediction)
        result = hangman.model.ml.utils.TO_CHAR[index]

        return result


def call_backs(ouput_path: str) -> List["tensorflow.keras.callbacks.ModelCheckpoint"]:
    """
    Create model call backs to store model weights after each epoch the
    loss is reduced

    :param ouput_path: (str) path to store
    :return: List["tensorflow.keras.callbacks.ModelCheckpoint"]
    """

    # define the checkpoint
    filepath = os.path.join(ouput_path, "lstm-{epoch:02d}-{loss:.4f}.keras")
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath, monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    return [checkpoint]
