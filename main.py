import numpy as np
from gensim.models.deprecated.keyedvectors import KeyedVectors
from gensim.scripts import glove2word2vec
from itertools import permutations
from hard_debias.word_embedding import WordEmbedding
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import json
import random


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


# target_words has pairs of sets X and Y which are words associated with concepts suspected to be biased
# towards one gender or another, i.e. Career vs. Family or science vs. arts
# returns a list of p-values that are the probability that a target-word pair is not biased
# higher p-values are less biased
def w2v_weat_eval(E, definitional_pairs, target_words):
    def moa(word, male, female):  # measure of association
        male_sum = 0
        female_sum = 0
        for mword in male:
            male_sum += cosine(E.v(word), E.v(mword))
        for fword in female:
            female_sum += cosine(E.v(word), E.v(fword))
        return ((1 / (len(male))) * male_sum) - ((1 / (len(female))) * female_sum)

    # takes definitional pairs and splits them into sets for each gender
    def build_gendered_lists(def_pairs):
        mwords = set()
        fwords = set()
        for pair in def_pairs:
            mwords.add(pair[0])
            fwords.add(pair[1])
        return mwords, fwords

    # finds the test statistic, which we use to obtain the probability that
    # the word embedding is biased for the given subject
    def find_test_statistic(pair, mwords, fwords):
        s1 = pair[0]  # subject one
        s2 = pair[1]  # subject two
        s1_sum = 0  # test statistic
        s2_sum = 0
        for word in s1:
            s1_sum += moa(word, mwords, fwords)
        for word in s2:
            s2_sum += moa(word, mwords, fwords)
        return s1_sum - s2_sum

    mwords, fwords = build_gendered_lists(definitional_pairs)
    p_values = []
    num_runs = 100
    for pair in target_words:
        test_statistic_greater = 0
        o_sum = find_test_statistic(pair, mwords, fwords)  # observed test statistic
        seen = set()
        union_set = pair[0] + pair[1]
        # for partition in partitions:
        for i in range(num_runs):
            perm = tuple(random.sample(union_set, len(union_set)))
            part = (perm[:len(perm) // 2], perm[len(perm) // 2:])
            if part not in seen:
                t_sum = find_test_statistic(part, mwords, fwords)
                print(part)
                print(o_sum, t_sum)
                if (o_sum < t_sum):
                    test_statistic_greater += 1
                seen.add(part)
        p_values.append(test_statistic_greater / num_runs) # temp for testing
        # p_values.append(test_statistic_greater / len(partitions))
    return p_values

def find_target_words(E, clusters):
    model = KMeans(n_clusters=clusters).fit(E)

    pass

if __name__ == "__main__":
    file_path = 'debiased_we/double_hd_we.txt'
    definitional_filename = 'hard_debias/data/definitional_pairs.json'

    with open(definitional_filename, "r") as f:
        defs = json.load(f)

    target_words = [(['executive', 'management', 'professional', 'corporation', 'salary'], ['home', 'parents', 'children', 'family', 'cousins']), (), ()]
    E = WordEmbedding(file_path)
    p_values = w2v_weat_eval(E, defs, target_words)
    print(p_values)
