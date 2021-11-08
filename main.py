import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec, KeyedVectors


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


def gensim_convert_golve_to_wrod2vec():
    print("Loading Glove Model in gensim_convert_golve_to_wrod2vec")
    glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    print(glove_model['hello'])
    return glove_model


if __name__ == "__main__":
    load_word_embedding()
    print("--------------------------------------------------------------")
    gensim_convert_golve_to_wrod2vec()
