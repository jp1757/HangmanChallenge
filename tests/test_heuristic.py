"""
Test Heuristic class
"""

import os
import sys

# insert project directory to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), "..")))

import unittest

import hangman.model.basic


class TestHeuristic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.words = ["hello", "hi", "yes", "please"]

    @classmethod
    def instance(cls):
        """Return an instance of Heuristic class"""
        return hangman.model.basic.Heuristic(cls.words)

    def test_guess(self):
        """Test expected guessed based on short dictionary of words"""

        word_masked = ["_____", "_e___", "_ell_", "hell_", "hello"]
        player = self.instance()
        guesses = ["e", "l", "h", "o"]

        for g, w in zip(guesses, word_masked):
            guess = player.guess(w)
            self.assertEqual(g, guess)
