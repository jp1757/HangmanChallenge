"""
Light-weight interface to return a model configuration object
"""

import abc
from typing import Any


class IConfig(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build(self) -> Any:
        """
        Return instance of built model
        :return: Instance of keras model that has a train & prediction function
        """
        pass

    @abc.abstractmethod
    def load(self, compile_model: bool) -> Any:
        """
        Load model from file

        :param compile_model: (bool) whether to comile model
        :return: Instance of keras model that has a train & prediction function
        """
        pass

    @abc.abstractmethod
    def load_weights(self, model: Any, compile_model: bool) -> Any:
        """
        Load weights from file into passed model

        :param model: keras LSTM model
        :param compile_model: (bool) whether to comile model
        :return: Instance of keras model that has a train & prediction function
        """
        pass
