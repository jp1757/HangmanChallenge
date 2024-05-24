"""
Light-weight interface for defining an object that returns
letter _guesses based on an input masked word
"""

import abc


class IPlayer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset player state to play a new game"""
        pass

    @abc.abstractmethod
    def guess(self, word: str) -> str:
        """
        Method for guessing letters based on input masked word

        :param word: masked word to guess letters in i.e "h_pp_" (starts fully masked)
        :return: (char) letter guess
        """
        pass
