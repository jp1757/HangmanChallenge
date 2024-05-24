"""Light-weight interface for defining ML models"""

import abc
from typing import Type


class IModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, epochs: int = 50, batch_size: int = 64):
        """Train model"""
        pass

    @abc.abstractmethod
    def predict(self, x: Type["np.array"]) -> str:
        """
        Predict y values based on pre-trained model

        :param x: (np.array) 3D array (# Samples, # Time Steps, # Features)
        :return: (str) prediction
        """
        pass
