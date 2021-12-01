import numpy as np
from utils import limit_vocab
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import extract_vectors
from utils import doPCA, drop
import scipy
import codecs, json
import operator

definitional_pairs = [['she', 'he'], ['herself', 'himself'], ['her', 'his'], ['daughter', 'son'],
                      ['girl', 'boy'], ['mother', 'father'], ['woman', 'man'], ['mary', 'john'],
                      ['gal', 'guy'], ['female', 'male']]


def load_glove(path):
    print("Loading word embedding...")
    with open(path) as f:
        lines = f.readlines()

    wv = []
    vocab = []
    for line in lines:
        tokens = line.strip().split(" ")
        assert len(tokens) == 301
        vocab.append(tokens[0])
        wv.append([float(elem) for elem in tokens[1:]])
    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.array(wv).astype(float)
    # n, d = wv.shape
    print(len(vocab), wv.shape, len(w2i))

    return wv, w2i, vocab


def simi(a, b):
    return 1 - scipy.spatial.distance.cosine(a, b)


def compute_bias_by_projection(wv, w2i, vocab, he_embed, she_embed):
    d = {}
    for w in vocab:
        u = wv[w2i[w], :]
        d[w] = simi(u, he_embed) - simi(u, she_embed)
    return d


# debias
# get main PCA components
def my_pca(wv):
    wv_mean = np.mean(np.array(wv), axis=0)
    wv_hat = np.zeros(wv.shape).astype(float)

    for i in range(len(wv)):
        wv_hat[i, :] = wv[i, :] - wv_mean

    main_pca = PCA()
    main_pca.fit(wv_hat)

    return main_pca


def hard_debias(wv, w2i, main_pca, wv_mean, w2i_partial, vocab_partial, component_ids):
    D = []

    for i in component_ids:
        D.append(main_pca.components_[i])

    # get rid of frequency features
    wv_f = np.zeros((len(vocab_partial), wv.shape[1])).astype(float)

    for i, w in enumerate(vocab_partial):
        u = wv[w2i[w], :]
        sub = np.zeros(u.shape).astype(float)
        for d in D:
            sub += np.dot(np.dot(np.transpose(d), u), d)
        wv_f[w2i_partial[w], :] = wv[w2i[w], :] - sub - wv_mean

    # debias
    gender_directions = list()
    for gender_word_list in [definitional_pairs]:
        gender_directions.append(doPCA(gender_word_list, wv_f, w2i_partial).components_[0])

    wv_debiased = np.zeros((len(vocab_partial), len(wv_f[0, :]))).astype(float)
    for i, w in enumerate(vocab_partial):
        u = wv_f[w2i_partial[w], :]
        for gender_direction in gender_directions:
            u = drop(u, gender_direction)
            wv_debiased[w2i_partial[w], :] = u

    return wv_debiased


if __name__ == '__main__':
    wv, w2i, vocab = load_glove('../embedding/w2v_gnews_small.txt')

    exclude_words = []
    with open('./data/male_word_file.txt') as f:
        for l in f:
            exclude_words.append(l.strip())
    with open('./data/female_word_file.txt') as f:
        for l in f:
            exclude_words.append(l.strip())

    with codecs.open('./data/gender_specific_full.json') as f:
        exclude_words.extend(json.load(f))

    definitional_words = []
    for pair in definitional_pairs:
        for word in pair:
            definitional_words.append(word)

    vocab_limit, wv_limit, w2i_limit = limit_vocab(wv, w2i, vocab, exclude=exclude_words)

    # compute original
    he_embed = wv[w2i['he'], :]
    she_embed = wv[w2i['she'], :]

    gender_bias_bef = compute_bias_by_projection(wv_limit, w2i_limit, vocab_limit, he_embed, she_embed)
    main_pca = my_pca(wv)
    wv_mean = np.mean(np.array(wv), axis=0)

    size = 1000
    sorted_g = sorted(gender_bias_bef.items(), key=operator.itemgetter(1))
    female = [item[0] for item in sorted_g[:size]]
    male = [item[0] for item in sorted_g[-size:]]

    c_vocab = list(set(male + female + [word for word in definitional_words if word in w2i]))
    c_w2i = dict()
    for idx, w in enumerate(c_vocab):
        c_w2i[w] = idx

    for component_id in range(20):
        wv_debiased = hard_debias(wv, w2i, main_pca, wv_mean, w2i_partial=c_w2i, vocab_partial=c_vocab,
                                  component_ids=[component_id])

    filename = '../debiased_we/double_hd_we.txt'
    print("size of vocab:", len(vocab))
    print("size of wv: ", len(wv))
    print("size of wv_debiased: ", len(wv_debiased))
    with open(filename, "w") as f:
        f.write("\n".join([w + " " + " ".join([str(x) for x in v]) for w, v in zip(vocab, wv_debiased)]))
    print("Wrote", len(wv_debiased), "words to", filename)
