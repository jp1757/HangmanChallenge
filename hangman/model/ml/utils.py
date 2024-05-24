"""
Utility module that has useful functions to support ML process.
Building training data etc
"""

import itertools
import json
import os
import string
from typing import List, Tuple, Type

import nltk
import numpy as np
import tensorflow.keras.preprocessing.sequence

import hangman.model.ml
import hangman.core.dictionary

MASKED_CHAR = "_"
TO_CHAR = {
    x + 1: y for x, y in tuple(enumerate(list(string.ascii_lowercase) + [MASKED_CHAR]))
}
TO_INT = {y: x for x, y in TO_CHAR.items()}


def mask_generator(word: str, *, min_letters: int = 2) -> List[str]:
    """
    Create all combinations of a word replacing 1 or n-1 letters with
    the masked_char where n = len(_word)
    
    :param word: (str) input word to create masked combinations over
    :param min_letters: (int) minimum actual letters to leave in each mask
        i.e. if word="hello" & min_letters=2, function will not create
        any combinations with < 2 of the original letters from the word
        "h____" would not be created

    :return: List[str] of all newly created combinations of masked words
    """

    # Basic variables
    word_len = len(word)
    indexes = list(range(0, word_len))
    _range = range(len(indexes) + 1)

    # Get all combinations of indices and lengths in word apart from () or all
    combos = [y for n in _range for y in list(itertools.combinations(indexes, n))][1:-1]

    # Create list of dictionaries where key is index in target word based on combos
    # generated above & value is masked_char
    _l = [
        {x: MASKED_CHAR for x in y}
        for y in combos if (word_len - len(y)) >= min_letters
    ]

    # Convert word into dict where key is index & value is letter
    # then duplicate into a list same size as combos
    _wd = dict(enumerate(word))
    _wd = [_wd] * len(combos)

    # Merge base dict of word with all letters to each newly masked
    # dict where some letters are replaced by masked_char
    new_words = ["".join({**x, **y}.values()) for x, y in zip(_wd, _l)]

    return new_words


def build_masks(words: List[str], mask_path: str, min: int = 3, max: int = 15) -> None:
    """
    Build all combinations of masked words for all words with lengths
    ranging between min & max

    :param words: List[str] all words to build masked combinations over
    :param mask_path: dense_units path to write masked combinations to. A new
        file will be created for each word length

    :return: None
    """

    # Get dataframe of words with length
    word_df = hangman.core.dictionary.dataframe(words=words)

    for _l in range(min, max + 1):
        with open(os.path.join(mask_path, f"{str(_l)}.txt"), 'w') as f:
            _words = list(word_df[word_df.len == _l].word)
            for _w in _words:
                f.write("\n".join(mask_generator(_w)))
                f.write("\n")


def n_gram(
        words: List[str],
        *,
        n_min: int = 2,
        n_max: int = 7,
        clean_mask: bool = False,
        reverse: bool = True,
) -> Tuple[Tuple[Tuple[str]], Tuple[str]]:
    """
    Create ngrams of size ranging between n_min & n_max, for all input words.

    :param words: words to calculate ngrams for
    :param n_min: min size of ngram
    :param n_max: max size of ngrams
    :param clean_mask: remove ngrams that are predicting the masked char
    :param reverse: add in all reversed ngrams

    :return: xy = Tuple[Tuple[Tuple[str]], Tuple[str]] where
        - x = Tuple of all ngrams [:-1] (Tuple of Tuples)
        - y = Tuple of all ngrams [-1] (single Tuple)
    """

    ngrams = [list(nltk.ngrams(x, y)) for x in words for y in range(n_min, n_max)]
    ngrams_unpac = [y for x in ngrams for y in x]  # unpack nested ngram lists
    ngrams_unique = list(set(ngrams_unpac))  # remove duplicates

    # Add ngrams in reverse
    if reverse:
        ngrams_unique = ngrams_unique + [tuple(reversed(x)) for x in ngrams_unique]

    xy = [(x[:-1], x[-1]) for x in ngrams_unique]

    # Remove ngrams that are predicting the masked_char or all masked_char(s)
    # are making the prediction i.e. either (('a,'b','c'), ('_')) or
    # (('_', '_'), ('a')) are both filtered out
    if clean_mask:
        xy = [
            x for x in xy
            if (x[1] != MASKED_CHAR) and (not (np.array(x[0]) == MASKED_CHAR).all())
        ]

    # Unzip into two tuples where [0] will be used to predict [1]
    x_char, y_char = zip(*xy)

    xchar_len = len(x_char)
    assert xchar_len == len(y_char)

    xchar_half = round(xchar_len / 2)
    assert xy[xchar_half][0] == x_char[xchar_half]
    assert xy[xchar_half][1] == y_char[xchar_half]

    return x_char, y_char


def build_ngrams(input_paths: List[str], output_path: str) -> None:
    """
    Build all ngrams for words loaded from input paths & dense_units them
    to json to target dense_units path.  Each json will be a dict with
    two keys:
        - 'x' = x_char return from n_gram function
        - 'y' = y_char return from n_gram function

    :param input_paths: list of files to load words from
    :param output_path: directory to write json files to. File names
        will match input file names changing to json extension

    :return: None
    """

    for path in input_paths:
        _path, _mfn = os.path.splitext(_mfn)

        print(f"Building ngrams for [{_mfn}]")

        # Load word file
        with open(path) as file:
            _words = [line.rstrip() for line in file]

        # Build ngrams for loaded file
        _ngrams = n_gram(_words, clean_mask=True)

        print(f"Calculated n=[{len(_ngrams[0])}] ngrams for [{_mfn}]")

        # Write ngrams to json
        _data_out = {"x": _ngrams[0], "y": _ngrams[1]}
        _mfn_name, _mfn_ext = os.path.splitext(_mfn)

        with open(os.path.join(output_path, f"{_mfn_name}.json"), 'w') as f:
            json.dump(_data_out, f)


def load_ngrams(directory: str) -> List[Tuple[Tuple[Tuple[str]], Tuple[str]]]:
    """
    Load ngrams from json files

    :param directory: directory containing jsons

    :return: List of Tuples
        - [1] - x_char from n_gram function concatenated across input files
        - [2] - y_char from n_gram function concatenated across input files
    """

    x_char, y_char = [], []
    ngram_path_files_path = os.listdir(directory)

    for _ngfn in ngram_path_files_path:

        if os.path.splitext(_ngfn)[1] != ".json":
            continue

        print(f"Loading ngrams from [{_ngfn}]")

        # Read ngrams from json
        with open(os.path.join(directory, _ngfn)) as f:
            _json = json.load(f)

        x_char = x_char + [tuple(x) for x in _json["x"]]
        y_char = y_char + _json["y"]

    assert len(x_char) == len(y_char)

    _xy = tuple(set(zip(x_char, y_char)))
    _xy = [z for z in _xy if not (np.array(z[0]) == MASKED_CHAR).all()]
    _x, _y = zip(*_xy)

    assert len(_x) == len(_y) == len(_xy)

    return _x, _y


def model_input(x_char: Tuple[Tuple[str]], y_char: Tuple[str]) -> Tuple[Type["np.array"]]:
    # Map input chars to ints
    x = [[hangman.model.ml.utils.TO_INT[xb] for xb in x] for x in x_char]
    y = [hangman.model.ml.utils.TO_INT[y] for y in y_char]

    x_len = len(x)
    assert x_len == len(y)
    assert x_len == len(x_char)

    x_half = round(x_len / 2)
    assert x[x_half] == [hangman.model.ml.utils.TO_INT[x] for x in x_char[x_half]]
    assert y[x_half] == hangman.model.ml.utils.TO_INT[y_char[x_half]]

    # Pad sequence to same left to the right
    x = tensorflow.keras.preprocessing.sequence.pad_sequences(x, padding="post")
    assert len(x) == x_len

    # Reshape to 3D Array for LSTM input (# Samples, # Time Steps, # Features)
    x = np.array(x).reshape(x_len, len(x[0]), 1)
    # Normalise
    x = x / len(hangman.model.ml.utils.TO_CHAR)
    y = tensorflow.keras.utils.to_categorical(y)

    return x, y
