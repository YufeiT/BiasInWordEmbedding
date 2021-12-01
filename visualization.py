import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pylab
from hard_debias.word_embedding import WordEmbedding

if __name__ == "__main__":
    we_file_path = "embedding/w2v_gnews_small.txt"
    hwe_file_path = "debiased_we/hd_em.txt"
    dhwe_file_path = "debiased_we/double_hd_we.txt"

    E = WordEmbedding(dhwe_file_path)
    X = E.get_matrix()
    print(np.shape(X))
    print(X[0])

    # pca = PCA(n_components=50)
    # reduced_x = pca.fit_transform(X)
    # print(np.shape(reduced_x))

    # Y = TSNE(n_components=2, perplexity=5.0, learning_rate='auto').fit_transform(reduced_x)
    # print(np.shape(Y))
    # pylab.scatter(Y[:, 0], Y[:, :1], 20)
    # pylab.show()



