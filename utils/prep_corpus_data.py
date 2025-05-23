import pandas as pd
from pyarabic import araby
import csv


def get_corpus_data(filename, old_path, new_path='analysis', idx=None, arabic=False):
    with open(old_path + filename + '.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("\t") for line in stripped if line)
        with open(new_path + filename + '.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)
    df = pd.read_csv(new_path + filename + '.csv', index_col=0, names=['word', 'frequency'])
    # remove symbols
    if idx is not None:
        df = df[idx:]
        df = df.reset_index()
        df = df.drop('index', axis=1)
    df = determine_length(df, arabic)
    df.to_csv(new_path + filename + '.csv', index=False)


def determine_length(df, arabic=False):
    """expects a dataframe with a column called 'word' that contains words the length of which should be determined"""
    if arabic:
        df["length"] = df["word"].apply(lambda x: len(araby.strip_diacritics(x)) if x.isalpha() and x.isascii() == False else 0)
    else:
        df["length"] = df["word"].apply(lambda x: len(x) if type(x) == str and x.isalpha() and x.isascii() else 0)
    # keep only words with at least one letter
    df = df[df["length"] > 0]
    return df

