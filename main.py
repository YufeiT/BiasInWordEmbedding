import numpy as np
import pandas as pd
import csv


def load_word_embedding():
    file_path = "glove.6B.300d.txt"
    print("Loading Glove Model")
    glove_model = {}
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    print(glove_model['hello'])
    return glove_model


if __name__ == "__main__":
    load_word_embedding()
