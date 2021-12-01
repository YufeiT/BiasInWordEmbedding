import numpy as np
from numpy import linalg as LA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from hard_debias.word_embedding import WordEmbedding
from scipy.spatial.distance import cosine
import operator


def merge_dhd_hd(E, DHE):
    dh_index = DHE.get_index()
    dh_vectors = DHE.get_matrix()

    final_matrix = E.get_matrix()
    h_index = E.get_index()

    for word in dh_index.keys():
        final_matrix[h_index[word]] = dh_vectors[dh_index[word]]
    return final_matrix, h_index


def normalize(wv):
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv


def simi(a, b):
    return 1 - cosine(a, b)


def compute_bias_by_projection(E):
    index = E.get_index()
    d = {}
    for w in index.keys():
        u = E.v(w)
        d[w] = simi(u, E.v('he')) - simi(u, E.v('she'))
    return d


def visualize(vectors, y_true, y_pred, ax, title, random_state):
    # perform TSNE
    vectors = normalize(vectors)
    X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(vectors)
    for x, p, y in zip(X_embedded, y_pred, y_true):
        if y:
            ax.scatter(x[0], x[1], marker='.', c='c')
        else:
            ax.scatter(x[0], x[1], marker='x', c='darkviolet')
    return ax


def cluster_and_visualize(words, X1, title, random_state, tsne_random_state, y_true, num=2):
    kmeans_1 = KMeans(n_clusters=num, random_state=random_state).fit(X1)
    y_pred_1 = kmeans_1.predict(X1)
    correct = [1 if item1 == item2 else 0 for (item1, item2) in zip(y_true, y_pred_1)]
    print('precision', max(sum(correct) / float(len(correct)), 1 - sum(correct) / float(len(correct))))

    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    ax1 = visualize(X1, y_true, y_pred_1, axs, title, tsne_random_state)

    fig.savefig("a_{}_{}_{}.png".format(title, size, random_state))


def extract_vectors(words, E):
    m = np.empty(shape=(1000, 300), dtype=float, order='C')
    for i, word in enumerate(words):
        m[i] = E.v(word)
    return m


if __name__ == "__main__":
    we_file_path = "embedding/w2v_gnews_small.txt"
    hwe_file_path = "debiased_we/hd_em.txt"
    dhwe_file_path = "debiased_we/double_hd_we.txt"

    E = WordEmbedding(we_file_path)
    gender_bias_bef = compute_bias_by_projection(E)
    sorted_g = sorted(gender_bias_bef.items(), key=operator.itemgetter(1))

    size = 500
    random_state = 0
    tsne_random_state = 2
    female = [item[0] for item in sorted_g[:size]]
    male = [item[0] for item in sorted_g[-size:]]
    y_true = [1] * size + [0] * size
    cluster_and_visualize(male + female, extract_vectors(male + female, E), 'Original w2v Google News', random_state, tsne_random_state, y_true)

    E = WordEmbedding(hwe_file_path)
    cluster_and_visualize(male + female, extract_vectors(male + female, E), 'Hard Debiased w2v Google News', random_state, tsne_random_state, y_true)

    DHE = WordEmbedding(dhwe_file_path)
    matrix, index = merge_dhd_hd(E, DHE)
    DHE.overwrite_index(index)
    DHE.overwrite_matrix(matrix)
    cluster_and_visualize(male + female, extract_vectors(male + female, DHE), 'Double Hard Debiased w2v Google News',
                          random_state, tsne_random_state, y_true)
