"""
Test Hangman games with various different models & strategies for selecting letters
"""

import os
import sys

# insert project directory to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), "..")))

import unittest
from typing import List, Set

import hangman.core
import hangman.core.api
import hangman.model


################################
###Dummy classes for testing####
################################
class APIDummy(hangman.core.IAPI):
    """Dummy API implementation"""

    def __init__(self, tries_remains: int, letters: List[str]):
        self.tries_remains = tries_remains
        self.letters = letters
        self._guesses = set()

    @property
    def game_state(self) -> hangman.core.api.Status:
        if (self.tries_remains > 0) and (len(self.letters) == len(self.guesses)):
            return hangman.core.api.Status.SUCCESS
        elif self.tries_remains > 0:
            return hangman.core.api.Status.ONGOING
        else:
            return hangman.core.api.Status.FAILED

    @property
    def letters_found(self) -> int:
        return 0

    @property
    def guesses(self) -> Set[str]:
        return self._guesses

    @property
    def word(self) -> str:
        """Return word based on current correct _guesses"""
        return ""

    def guess(self, char: str) -> hangman.core.api.Response:

        if self.game_state == hangman.core.api.Status.FAILED:
            return hangman.core.api.Response(
                word=self.word, status=self.game_state, correct_guess=False
            )

        correct = char in self.letters
        if correct:
            self._guesses.update(char)
        else:
            self.tries_remains = self.tries_remains - 1

        return hangman.core.api.Response(
            word="", status=self.game_state, correct_guess=correct
        )


class PlayerDummy(hangman.model.IPlayer):
    """Dummy Guesser class for selecting letters"""

    def __init__(self, letters: List[str]):
        self.letters = letters
        self.__idx = -1

    def reset(self) -> None:
        self.__idx = -1

    def guess(self, word: str) -> str:
        self.__idx = self.__idx + 1
        return self.letters[self.__idx]


################################
###Test Classes####
################################
class TestHangman(unittest.TestCase):

    def test_win(self):
        """Test basic game that wins"""

        # Init dummy variables
        letters = ["a", "b", "c"]
        api = APIDummy(4, letters)
        player = PlayerDummy(letters)

        # Init game instance
        game = hangman.core.Hangman(api=api, player=player)

        # Play game
        status = game.start_game(verbose=False)

        self.assertTrue(status.win)

    def test_win_mix(self):
        """Test basic game that wins with some incorrect _guesses"""

        # Init dummy variables
        api = APIDummy(4, ["a", "b", "c"])
        player = PlayerDummy(["d", "a", "e", "b", "c"])

        # Init game instance
        game = hangman.core.Hangman(api=api, player=player)

        # Play game
        status = game.start_game(verbose=False)

        self.assertTrue(status.win)

    def test_loss(self):
        """Test basic game that loses"""

        # Init dummy variables
        api = APIDummy(4, ["a", "b", "c"])
        player = PlayerDummy(["d", "e", "f", "g"])

        # Init game instance
        game = hangman.core.Hangman(api=api, player=player)

        # Play game
        status = game.start_game(verbose=False)

        self.assertFalse(status.win)

    def test_loss(self):
        """Test basic game that loses with some correct _guesses"""

        # Init dummy variables
        api = APIDummy(4, ["a", "b", "c"])
        player = PlayerDummy(["a", "b", "d", "e", "f", "g"])

        # Init game instance
        game = hangman.core.Hangman(api=api, player=player)

        # Play game
        status = game.start_game(verbose=False)

        self.assertFalse(status.win)
