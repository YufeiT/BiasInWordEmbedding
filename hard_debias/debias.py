import word_embedding
import json
import numpy as np


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = word_embedding.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = word_embedding.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if a in E.index and b in E.index:
            y = word_embedding.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y) ** 2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":
    embedding_filename = '../embedding/w2v_gnews_small.txt'
    definitional_filename = 'data/definitional_pairs.json'
    gendered_words_filename = 'data/gender_specific_full.json'
    equalize_filename = 'data/equalize_pairs.json'
    debiased_filename = 'debiased_we/debiased_em.txt'

    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = word_embedding.WordEmbedding(embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(debiased_filename)
    else:
        E.save(debiased_filename)

    print("\n\nDone!\n")
