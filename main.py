import numpy as np
from gensim.models.deprecated.keyedvectors import KeyedVectors
from gensim.scripts import glove2word2vec
from itertools import permutations


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
            male_sum += np.cos(E.v(word), E.v(mword))
        for fword in female:
            female_sum += np.cos(E.v(word), E.v(fword))
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
        return s1_sum, s2_sum

    # creates permutations of the union set where
    def create_partitions(union_set):
        permutation = permutations(union_set)
        part = list()
        for perm in permutation:
            part.append((perm[:len(perm) // 2], perm[len(perm) // 2:]))
        return part

    mwords, fwords = build_gendered_lists(definitional_pairs)
    p_values = []
    for pair in target_words:
        test_statistic_greater = 0
        o_sum = find_test_statistic(pair, mwords, fwords)  # observed test statistic
        partitions = create_partitions(pair[1].union(pair[0]))
        for partition in partitions:
            t_sum = find_test_statistic(partition, mwords, fwords)
            if (o_sum < t_sum):
                test_statistic_greater += 1
        p_values.append(test_statistic_greater / len(partitions))
    return p_values



if __name__ == "__main__":
    # load_word_embedding()
    print("--------------------------------------------------------------")
    # gensim_convert_golve_to_wrod2vec()
    import_em()
