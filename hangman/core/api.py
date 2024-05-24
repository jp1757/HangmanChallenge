"""
A class that simulates Trexquant API.
"""

import abc
from dataclasses import dataclass
from enum import Enum
from typing import List, Set

import numpy as np


class Status(Enum):
    SUCCESS = 1
    FAILED = 2
    ONGOING = 3


@dataclass
class Response:
    word: str
    status: Status
    correct_guess: bool


class IAPI(metaclass=abc.ABCMeta):
    """Light-weight interface for defining an API"""

    @property
    @abc.abstractmethod
    def game_state(self) -> Status:
        """Returns the current game state"""
        pass

    @property
    @abc.abstractmethod
    def word(self) -> str:
        """Return word based on current correct _guesses"""
        pass

    @property
    @abc.abstractmethod
    def letters_found(self) -> int:
        """Return number of letters found"""
        pass

    @property
    @abc.abstractmethod
    def guesses(self) -> Set[str]:
        """Returns list of all _guesses"""
        pass

    @abc.abstractmethod
    def guess(self, char: str) -> Response:
        """
        Check a letter guess against stored word. If present return
        masked word with that letter revealed at all indexes.

        :param char: (str) letter to check

        :return: (api.Response)
            - api.Response.word (str) current state of masked word
              with successful letter _guesses inserted
            - api.Response.status (api.Status) current game state
              (options: SUCCESS, FAILED, ONGOING)
        """
        pass


class API(IAPI):
    """
    Generates random word & checks letter _guesses
    """

    def __init__(
            self,
            dictionary: List[str],
            *,
            max_tries: int = 6,
            word: str = None,
            mask_char: str = "_",
            verbose: bool = False,
    ) -> None:
        """
        Init new api dummy object

        :param dictionary: (List[str]) of input words to select from
        :param max_tries: (int) (default=6) max number of letter _guesses allowed
        :param word: (str) (default=None) override random word choice (useful for testing)
        :param mask_char: (str) to use as mask character
        :param verbose: (bool) when True print messages to console
        """

        self.dictionary = dictionary
        self.__word = np.random.choice(self.dictionary) if word is None else word

        self._current_word = "".join([mask_char] * len(self.__word))
        self._current_dict = dict(enumerate(self._current_word))

        # store sets of correct & incorrect _guesses
        self._valid = set()
        self._invalid = set()

        self.max_tries = max_tries
        self.tries_remains = max_tries
        self.num_tries = 0

        self.verbose = verbose

    @property
    def game_state(self) -> Status:
        """Returns the current game state"""

        if (self.tries_remains > 0) and (self.__word == self.word):
            return Status.SUCCESS
        elif self.tries_remains > 0:
            return Status.ONGOING
        else:
            return Status.FAILED

    @property
    def guesses(self) -> Set[str]:
        """Returns list of all _guesses"""
        return self._valid.union(self._invalid)

    @property
    def letters_found(self) -> int:
        """Return number of letters found"""
        return len(self._valid)

    @property
    def word(self) -> str:
        """Return word based on current correct _guesses"""
        return self._current_word

    def guess(self, char: str) -> Response:
        """
        Check a letter guess against stored word. If present return
        masked word with that letter revealed at all indexes.

        :param char: (str) letter to check

        :return: (api.Response)
            - api.Response.word (str) current state of masked word
              with successful letter _guesses inserted
            - api.Response.status (api.Status) current game state
              (options: SUCCESS, FAILED, ONGOING)
        """

        if not len(char) == 1:
            raise ValueError(f"Only guess one character at a time")
        elif self.game_state == Status.FAILED:
            return Response(word=self.word, status=self.game_state, correct_guess=False)

        # Update num tries
        self.num_tries = self.num_tries + 1

        # Get all indexes of char in word
        _valid = {i: char for i, c in enumerate(self.__word) if c == char}

        # Correct guess
        correct_guess = False
        if len(_valid) > 0:
            # Update dict containing valid _guesses
            self._valid.update(char)
            self._current_dict = {**self._current_dict, **_valid}
            self._current_word = "".join(
                [x[1] for x in sorted(self._current_dict.items())]
            )
            correct_guess = True
            if self.verbose:
                print(f"Correct guess: [{char}]")

        # Incorrect Guess
        else:
            self._invalid.update(char)
            self.tries_remains = self.tries_remains - 1

            if self.verbose:
                print(f"Incorrect guess: [{char}]")

        _state = self.game_state
        return Response(
            word=self.__word if _state == Status.FAILED else self.word,
            status=_state,
            correct_guess=correct_guess
        )
