import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec, KeyedVectors


def gensim_convert_golve_to_wrod2vec():
    print("Loading Glove Model in gensim_convert_golve_to_wrod2vec")
    # glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    # print(glove_model['hello'])
    print(glove_model.most_similar(positive=['woman', 'king'], negative=['man']))
    return glove_model


def import_em():
    model = KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)
    model.save_word2vec_format('embedding/GoogleNews-vectors-negative300.txt', binary=False)
    print(model['nurse'])


if __name__ == "__main__":
    # load_word_embedding()
    print("--------------------------------------------------------------")
    # gensim_convert_golve_to_wrod2vec()
    import_em()
