"""Data utility functions"""

import os
from typing import List

import pandas as pd

DATA = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", "data"))
WORDS = os.path.join(DATA, "words_250000_train.txt")


def dataframe(words: List[str]) -> pd.DataFrame:
    """
    Convert list of words into a pandas dataframe with 2 columns.
    First with word named 'word' second with word length name 'len'

    :param: List[str] list of words to convert to dataframe

    :return pd.DataFrame with 2 columns [word, len]
    """

    df = pd.DataFrame(words)
    df.rename(columns={0: "word"}, inplace=True)
    df["len"] = df["word"].str.len()
    df["word"] = df["word"].str.lower()

    midpoint = round(len(df) / 2)
    assert len(df.iloc[midpoint]["word"]) == df.iloc[midpoint]["len"]

    return df


def load(path: str = WORDS) -> List[str]:
    """
    Load word list from file

    :param path: file path of word list
    :return: List[str] of words
    """

    with open(path) as file:
        return [line.rstrip() for line in file]
