import stanza
import itertools
import os

nlp = stanza.Pipeline(lang='en', tokenize_no_ssplit=True)
depparse = nlp.processors['depparse']._trainer

import arbori
def GetKBest(weights, k):
    V = set(range(len(weights)))
    E = set()
    scores = {}
    for i in range(len(weights)):
        for j in range(len(weights)):
            e = arbori.Edge(j, i)
            E.add(e)
            scores[e] = weights[i, j]
        
    return arbori.GetKBest(k, V, E, scores)

def print_alternatives(doc, n_trees):
    head, deps = depparse.last_preds
    for i, sentence in enumerate(doc.sentences):
        for n_tree, (tree, score) in enumerate(GetKBest(head[i], n_trees)):
            words = list(sentence.words)
            print("{}Tree [{}/{}] (score={}){}".format("-"*5, n_tree + 1, n_trees, score, "-"*5))
            for j, word in enumerate(words):
                best_in_edge = tree[j + 1]
                source = best_in_edge.u - 1
                source_text = "ROOT" if source == -1 else words[source].text
                edge = depparse.vocab['deprel'].unmap((deps[i][j + 1][source + 1],))[0]
                print("{}->{}: {}".format(source_text, words[j].text, edge))
