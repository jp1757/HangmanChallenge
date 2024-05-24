"""Test ML utils"""

import os
import sys

# insert project directory to PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), "..")))

import unittest

import hangman.model.ml.utils


class TestMLUtils(unittest.TestCase):

    def test_mask_generator(self):
        """
        Test that all combinations of each word replacing with a
        mask char are generated correctly
        """

        self.assertListEqual(
            sorted(
                hangman.model.ml.utils.mask_generator("abcd", min_letters=0)
            ),
            ['___d', '__c_', '__cd', '_b__', '_b_d', '_bc_', '_bcd', 'a___',
             'a__d', 'a_c_', 'a_cd', 'ab__', 'ab_d', 'abc_']
        )

        self.assertListEqual(
            sorted(hangman.model.ml.utils.mask_generator("abcd", min_letters=2)),
            ['__cd', '_b_d', '_bc_', '_bcd', 'a__d', 'a_c_', 'a_cd', 'ab__', 'ab_d', 'abc_']
        )

        self.assertListEqual(
            sorted(hangman.model.ml.utils.mask_generator("abc", min_letters=0)),
            ['__c', '_b_', '_bc', 'a__', 'a_c', 'ab_']
        )

    def test_ngrams(self):
        """Test that ngrams of a range of sizes are created correctly for input words"""

        abc_x, abc_y = hangman.model.ml.utils.n_gram(["abc"])
        self.assertListEqual(
            sorted(list(zip(abc_x, abc_y))),
            [(('a',), 'b'),
             (('a', 'b'), 'c'),
             (('b',), 'a'),
             (('b',), 'c'),
             (('c',), 'b'),
             (('c', 'b'), 'a')]
        )

        abcd_x, abcd_y = hangman.model.ml.utils.n_gram(["abcd"])
        self.assertListEqual(
            sorted(list(zip(abcd_x, abcd_y))),
            [(('a',), 'b'),
             (('a', 'b'), 'c'),
             (('a', 'b', 'c'), 'd'),
             (('b',), 'a'),
             (('b',), 'c'),
             (('b', 'c'), 'd'),
             (('c',), 'b'),
             (('c',), 'd'),
             (('c', 'b'), 'a'),
             (('d',), 'c'),
             (('d', 'c'), 'b'),
             (('d', 'c', 'b'), 'a')]
        )

        comb_x, comb_y = hangman.model.ml.utils.n_gram(["abc", "abcd"])
        self.assertListEqual(
            sorted(list(set((list(zip(abc_x, abc_y)) + list(zip(abcd_x, abcd_y)))))),
            sorted(list(zip(comb_x, comb_y)))
        )

        _x, _y = hangman.model.ml.utils.n_gram(["ab_"], clean_mask=True)
        self.assertListEqual(
            sorted(list(zip(_x, _y))),
            [(('_', 'b'), 'a'), (('a',), 'b'), (('b',), 'a')]
        )
