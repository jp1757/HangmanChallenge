"""
Test API class
"""

import os
import sys

# insert project directory to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), "..")))

import unittest

import hangman.core.api


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.max_tries = 4
        cls.word = "hello"
        cls.mask_char = "+"

    @classmethod
    def instance(cls):
        """Return an instance of API class"""
        return hangman.core.API(
            dictionary=[], word="hello", max_tries=cls.max_tries, mask_char=cls.mask_char
        )

    def test_all_incorrect_fail(self):
        """Test all incorrect _guesses & game fails"""

        api = self.instance()

        for x in ["a", "b", "c", "d"]:
            guess = api.guess(x)

        self.assertEqual(guess.word, self.word)
        self.assertEqual(guess.status, hangman.core.api.Status.FAILED)
        self.assertEqual(api.word, "".join(len(self.word) * self.mask_char))
        self.assertEqual(api.game_state, hangman.core.api.Status.FAILED)
        self.assertEqual(api.num_tries, 4)
        self.assertEqual(api.tries_remains, 0)

    def test_all_incorrect_ongoing(self):
        """Test all incorrect _guesses & game ongoing"""

        api = self.instance()

        for x in ["a", "b", "c"]:
            guess = api.guess(x)

        self.assertEqual(guess.word, len(self.word) * self.mask_char)
        self.assertEqual(guess.status, hangman.core.api.Status.ONGOING)
        self.assertEqual(api.word, len(self.word) * self.mask_char)
        self.assertEqual(api.game_state, hangman.core.api.Status.ONGOING)
        self.assertEqual(api.num_tries, 3)
        self.assertEqual(api.tries_remains, 1)

    def test_all_correct(self):
        """Test all correct & wins game"""

        api = self.instance()

        _letters = ["h", "e", "l", "o"]
        _words = ["h", "he", "hell", "hello"]
        _states = [3, 3, 3, 1]

        for x, y, z in zip(_letters, _words, _states):
            guess = api.guess(x)

            self.assertEqual(
                guess.word, "".join([y] + (len(self.word) - len(y)) * [self.mask_char])
            )
            self.assertEqual(guess.status, hangman.core.api.Status(z))

        self.assertEqual(api.word, self.word)
        self.assertEqual(api.game_state, hangman.core.api.Status.SUCCESS)
        self.assertEqual(api.num_tries, 4)
        self.assertEqual(api.tries_remains, 4)

    def test_mix_success(self):
        """Test a mix of correct & incorrect _guesses with a game win"""

        api = self.instance()

        _letters = ["h", "e", "a", "b", "c", "l", "o"]
        _words = ["h", "he", "he", "he", "he", "hell", "hello"]
        _states = [3, 3, 3, 3, 3, 3, 1]

        for x, y, z in zip(_letters, _words, _states):
            guess = api.guess(x)

            self.assertEqual(
                guess.word, "".join([y] + (len(self.word) - len(y)) * [self.mask_char])
            )
            self.assertEqual(guess.status, hangman.core.api.Status(z))

        self.assertEqual(api.word, self.word)
        self.assertEqual(api.game_state, hangman.core.api.Status.SUCCESS)
        self.assertEqual(api.num_tries, 7)
        self.assertEqual(api.tries_remains, 1)

    def test_mix_fail(self):
        """Test a mix of correct & incorrect _guesses with a game fail"""

        api = self.instance()

        _letters = ["h", "e", "a", "b", "c", "d", "l", "o"]
        _words = ["h", "he", "he", "he", "he", "hello", "he", "he"]
        _states = [3, 3, 3, 3, 3, 2, 2, 2]

        for x, y, z in zip(_letters, _words, _states):
            guess = api.guess(x)

            self.assertEqual(
                guess.word, "".join([y] + (len(self.word) - len(y)) * [self.mask_char])
            )
            self.assertEqual(guess.status, hangman.core.api.Status(z))

        self.assertEqual(api.game_state, hangman.core.api.Status.FAILED)
        self.assertEqual(api.num_tries, 6)
        self.assertEqual(api.tries_remains, 0)
