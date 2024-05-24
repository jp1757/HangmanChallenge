"""
Defines a dual layer bi-directional LSTM model
"""
import os
from typing import Any, Type, Tuple

import tensorflow

import hangman.core.dictionary
from hangman.model.ml.config.iconfig import IConfig

MODEL_SPEC = os.path.join(hangman.core.dictionary.DATA, "lstm-dual-model.keras")
WEIGHTS = os.path.join(hangman.core.dictionary.DATA, "lstm-dual-weights.keras")

LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
DROPOUT = 0.2
LSTM_UNITS = 256


class DualBiDir(IConfig):

    def __init__(
            self,
            *,
            input: Tuple[int, int] = None,
            dense_units: int = None,
            lstm_units: int = LSTM_UNITS,
            drop_out: float = DROPOUT,
            loss: str = LOSS,
            optimizer: str = OPTIMIZER,
            model_path: str = MODEL_SPEC,
            weights_path: str = WEIGHTS,
    ) -> None:
        """
        Create instance of LSTM model

        :param input: Tuple(int, int) shape of input data
            see tensorflow.keras.Input docs
        :param dense_units: (int) dimensionality of output data
        :param lstm_units: +ve (int), dimensionality of the output space
        :param drop_out: (float) fraction of input units to drop
        :param loss: (str) loss function
        :param optimizer: (str) name of optimizer
        :param model_path: (str) path to load '.keras' model config from
        :param weights_path: (str) path to load '.keras' weights file from
        """

        # Store instance variables
        self.input = input
        self.dense_units = dense_units
        self.lstm_units = lstm_units
        self.drop_out = drop_out
        self.loss = loss
        self.optimizer = optimizer
        self.model_path = model_path
        self.weights_path = weights_path

    def build(self) -> Type["tensorflow.keras.models.Sequential"]:
        """Return instance of Keras Sequential model"""

        model = tensorflow.keras.models.Sequential()
        model.add(tensorflow.keras.Input(shape=self.input))
        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.LSTM(self.lstm_units, return_sequences=True)
            )
        )
        model.add(tensorflow.keras.layers.Dropout(self.drop_out))
        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.LSTM(self.lstm_units)
            )
        )
        model.add(tensorflow.keras.layers.Dropout(self.drop_out))
        model.add(tensorflow.keras.layers.Dense(self.dense_units, activation='softmax'))

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def load(self, compile_model: bool) -> Any:
        """
        Load model from file

        :param compile_model: (bool) whether to comile model
        :return: Instance of keras model that has a train & prediction function
        """
        model = tensorflow.keras.models.load_model(self.model_path)
        if compile_model:
            model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def load_weights(self, model: Any, compile_model: bool) -> Any:
        """
        Load weights from file into passed model

        :param model: keras LSTM model
        :param compile_model: (bool) whether to comile model
        :return: Instance of keras model that has a train & prediction function
        """
        model.load_weights(self.weights_path)
        if compile_model:
            model.compile(loss=self.loss, optimizer=self.optimizer)

        return model
