"""
Defines a heuristic approach to guessing letters
"""

import warnings
from collections import Counter
from typing import List

import numpy as np

from hangman.model import IPlayer


class Heuristic(IPlayer):
    """
    Guesses letter in a word using a basic heuristic approach
    of selecting the most frequent letter found across a set of
    words of equivalent length from an input dictionary
    """

    def __init__(self, dictionary: List[str]) -> None:
        """
        Create instance variables

        :param dictionary: List[str] of input words to use to formulate _guesses
        """

        self.dictionary = dictionary
        self._words = self.dictionary.copy()

        self._word = None  # Previous word state
        self._last = None  # Previous guess

        self.guesses = set()
        self.valid = set()

    def reset(self) -> None:
        """Reset player state to play a new game"""
        self._words = self.dictionary.copy()

        self._word = None  # Previous word state
        self._last = None  # Previous guess

        self.guesses = set()
        self.valid = set()

    def _validate(self, word: str) -> bool:
        """
        Checks latest guess against new word passed.  The word might not
        have changed which is interrpreted as an incorrec guess.  If the
        word has changed this is assumed to be a correct guess, as long as
        the last guess is also present in the word.  It returns a bool
        indicating whether the word has changed or not

        :param word: (str) masked word received from API
        :return: bool indicating whether word has changed
        """

        word_changed = False

        if self._word is not None:

            word_changed = self._word != word
            # Assume previous guess was correct
            if word_changed and (self._last in word):
                # Update valid _guesses set
                self.valid.update(self._last)

                # Get all indexes of char in word
                _valid = {i: c for i, c in enumerate(word) if c == self._last}

                # Update word list leaving only words that have the current guess
                # in matching position(s)
                self._words = [
                    x for x in self._words
                    if (self._last in x) and
                       (
                           np.array(
                               [
                                   x[idx] == self._last for idx, _x in _valid.items()
                                   if idx < len(x) - 1
                               ]
                           ).all()
                       )
                ]

                # Reset last guess so not checked again
                self._last = None

            # Assume incorrect Guess
            elif self._last is not None:
                self._words = [x for x in self._words if self._last not in x]

        self._word = word

        return word_changed

    def _update(self, letter: str) -> None:
        """Update instance collections"""

        if letter in self.guesses:
            warnings.warn(f"New guess [{letter}] already guessed: [{self.guesses}]")
        else:
            self.guesses.update(letter)
            self._last = letter

    def guess(self, word: str) -> str:
        """
        Method for guessing letters based on input masked word

        :param word: masked word to guess letters in i.e "h_pp_" (starts fully masked)
        :return: (char) letter guess
        """

        # Checks latest guess against input masked word
        self._validate(word)

        # Most frequent letter
        letters = ''.join(self._words)
        frequency = Counter(letters)

        new_guess = [x[0] for x in frequency.most_common() if x[0] not in self.guesses]
        if len(new_guess) == 0:
            # Reset words as we've run out of _guesses
            self._words = self.dictionary.copy()
            # Try again
            return self.guess(word)
        else:
            new_guess = new_guess[0]

        self._update(new_guess)

        return new_guess
