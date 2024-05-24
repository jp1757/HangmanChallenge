"""
Defines a game object similar to the one in the
Trexquant Jupyter Notebook - use for practice
away from Trexquant API
"""

from dataclasses import dataclass
from typing import Type

import hangman.core.api


@dataclass
class Response:
    win: bool
    guess_map: dict
    word: str
    guesses: set
    num_guesses: int


class Hangman:

    def __init__(self, *, api: Type["lib.core.API"], player: Type["lib.model.IPlayer"]) -> None:
        """
        Init game object to run a Hangman game using
            - an API object that randomly picks words and validates letter _guesses
            - a IPlayer object that _guesses letters based on a masked word

        :param api: (core.API) object that simulates Trexquant API & selects a random
            word from input dictionary then validates letters _guesses against that word
        :param player: (model.IPlayer) object to generate letters in an attempt to guess
            the target hidden word chosen by the API
        """

        self.api = api
        self.player = player

    def start_game(self, *, verbose=True) -> Response:
        """
        Generate a new random word from dictionary

        :return: (Response) Game Win (True) or Loss (False) + dict containing
            breakdown of num _guesses per letter
        """

        # Reset player to starting state
        self.player.reset()

        # Get word from api
        word = self.api.word

        if verbose:
            print(
                f"Starting word is [{len(word)}] characters, "
                f"[{self.api.max_tries}] lives remaining"
            )

        # Guess counter per letter
        word_len = len(word)
        letters_left = word_len
        guess_map = {letters_left: 1}

        while self.api.tries_remains > 0:

            # Get next letter guess
            guess_letter = self.player.guess(word)

            if verbose:
                print(f"Try # [{self.api.num_tries + 1}], guessing letter: {guess_letter}")

            # Check letter against API
            response = self.api.guess(guess_letter)

            # Update # _guesses per letter
            guess_count = guess_map.get(letters_left, None)
            if guess_count is None:
                guess_map.update({letters_left: 1})
            else:
                guess_map[letters_left] = guess_count + 1

            # Get letters left to guess
            if response.correct_guess:
                letters_left = word_len - self.api.letters_found

            # Get word from api
            word = response.word

            if verbose:
                print(
                    f"Guess [{'correct' if response.correct_guess else 'incorrect'}]. "
                    f"Word: [{word}]. "
                    f"Status: {response.status.name}. "
                    f"[{self.api.tries_remains}] lives remaining"
                )

            # Check API response
            if response.status == hangman.core.api.Status.SUCCESS:
                return Response(
                    win=True,
                    guess_map=guess_map,
                    word=word,
                    guesses=self.api._guesses,
                    num_guesses=len(self.api._guesses)
                )

            elif response.status == hangman.core.api.Status.FAILED:
                return Response(
                    win=False,
                    guess_map=guess_map,
                    word=word,
                    guesses=self.api._guesses,
                    num_guesses=len(self.api._guesses)
                )
