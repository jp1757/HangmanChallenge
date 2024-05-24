"""
Class containing an implementation of IPlayer interface
that defines logic for a player that uses neural network
to drive guess choices
"""

from collections import Counter, deque
from typing import List

import numpy as np

import hangman.model.ml.utils
from hangman.model.basic.heuristic import Heuristic
from hangman.model.ml.imodel import IModel


class NNPlayer(Heuristic):
    """
    Defines a class that uses a combination of ML & a more heuristic
    approach to select _guesses for the hangman game.

    It will strive to leverage the ML approach apart from at the start
    of the game when the word is fully-masked or the ML algorithm has
    run out of _guesses
    """

    def __init__(
            self,
            dictionary: List[str],
            *,
            model: IModel,
            verbose: bool = False,
            heuristic_thershold: float = 0.5,
    ) -> None:
        """
        Create instance variables & instantiate base class

        :param dictionary: List[str] of input words to use to formulate _guesses
        :param model: (hangman.model.ml.imodel.IModel) to use to generate ML driven _guesses
        :param verbose: (bool) when True prints out the source of the guess to std out
        """
        super().__init__(dictionary)

        self.model = model
        self._ml_guesses = deque()
        self.verbose = verbose
        self.heuristic_thershold = heuristic_thershold

    def reset(self) -> None:
        """Reset player state to play a new game"""
        super().reset() # Reset base

        self._ml_guesses = deque()

    def _guess(self, word_masked):

        # Create all n_grams to pass to model
        pred_x, pred_y = hangman.model.ml.utils.n_gram([word_masked])

        # Remove any occurrences that are made up of only the masked character
        pred = [
            x for x in set(pred_x)
            if not (np.array(x) == hangman.model.ml.utils.MASKED_CHAR).all()
        ]

        # Get the model prediction
        outputs = [self.model.predict(p) for p in pred]
        # Filter out anything that has already been _guesses
        outputs_f = [x for x in outputs if x not in self.guesses]

        # Return the _guesses in the order of what has been predicted the most
        return [x[0] for x in Counter(outputs_f).most_common()]

    def guess(self, word: str) -> str:
        """
        Method for guessing letters based on input masked word

        :param word: masked word to guess letters in i.e "h_pp_" (starts fully masked)
        :return: (char) letter guess
        """

        new_guess = None
        guess_type = ""

        fraction_left = 1 - sum([1 for x in word if x == hangman.model.ml.utils.MASKED_CHAR]) / len(word)

        # If no letters guessed correctly yet use heuristic guess
        if fraction_left >= self.heuristic_thershold:
            new_guess = super().guess(word)
            guess_type = "heuristic"

        # Try using ML guess
        else:
            # Checks latest guess against input masked word. True indicates word
            # has changed since last call
            if super()._validate(word):
                # Most frequent letter
                letters = ''.join(self._words)
                frequency = Counter(letters)
                most_frequent = set([x for x, y in frequency.most_common()[:3]])

                # Get ML guess(es)
                ml_guesses = self._guess(word)

                # Take intersection between two
                intersect = most_frequent.intersection(set(ml_guesses))

                self._ml_guesses = deque(intersect)

            # If there are ML _guesses use that otherwise revert to heuristic guess
            if len(self._ml_guesses) > 0:
                new_guess = self._ml_guesses.popleft()
                self._update(new_guess)
                guess_type = "ml"
            else:
                new_guess = super().guess(word)
                guess_type = "heuristic"

        if self.verbose:
            print(f"Guess source: [{guess_type}]")

        return new_guess
